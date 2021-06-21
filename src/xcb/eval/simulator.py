"""This module will have the XCB simulated bandit feedback evaluation."""
import os
import shutil
import gc
import logging
import scipy.sparse as smat
import numpy as np
from pathlib import Path
from tqdm import tqdm
import copy

from xcb.xfalcon import inference as xcb_inference
from xcb.xfalcon import train as xcb_train
from pecos.utils import smat_util

LOGGER = logging.getLogger(__name__)


class XCBSimulatedBandit(object):
    """Main class for simulated bandit evaluation."""

    def __init__(self, init_args, eval_args, train_config, inference_config, tmp_dir="/tmp"):
        """Init class.

        Parameters:
        ----------
        init_args: dict
            Models are initialized on this dataset. Dict keys are as follows:
            "X": smat.csr_matrix having init training features
            "Y": smat.csr_matrix having init target variables
        eval_args: dict
            Models are evaluated on this dataset. Dict Keys are as follows:
            "X": smat.csr_matrix having init training data
            "Y": smat.csr_matrix having init target variables
            "schedule": "exponential" or "linear" epoch schedule
            "batch_size": required if schedule if "linear".
        train_config: dict
            training parameters for the models, with keys:
            "mode": "ranker" or "full"
            "model_config":  configuration for model training
            "cluster_config": configuration for hierarchical clustering
            inference_config: dict
            config for inference with keys:
            "mode": currently supported "greedy" or "falcon" or "oracle" or "boltzmann"
            other keys are method dependent
        """

        self.init_args = init_args
        self.eval_args = eval_args
        self.train_config = train_config
        self.inference_config = inference_config
        self.tmp_dir = tmp_dir
        self.schedule = self._create_schedule()
        self.model_path = Path(self.tmp_dir, "model")
        self.current_cmat = None  # stores cluster chains
        if os.path.exists(self.model_path):
            shutil.rmtree(self.model_path)
        self._initialize_model()

    def evaluate(self):
        """Main function for evaluation."""
        rewards = np.array([0.0])
        for epoch in tqdm(range(len(self.schedule) - 1)):
            LOGGER.info("Starting epoch: {}".format(epoch))
            if epoch != 0:
                LOGGER.info("Training model for epoch {}".format(epoch))
                model = xcb_train.XFalconTrainer.train(
                    self.X_all,
                    self.Ys_all,
                    self.Ya_all,
                    self.Z_all,
                    self.train_config,
                    self.model_path,
                )
                LOGGER.info("Finished training model")
                shutil.rmtree(self.model_path)
                model.save(self.model_path)
                del model
            gc.collect()
            inf_obj, inf_config = self._get_inference_object()
            X_epoch = self.eval_args["X"][self.schedule[epoch] : self.schedule[epoch + 1], :]
            Y_epoch = self.eval_args["Y"][self.schedule[epoch] : self.schedule[epoch + 1], :]
            inf_config["X"] = smat.csr_matrix(X_epoch, dtype=np.float32)
            pred_Y_epoch, pred_maps = inf_obj.predict(**inf_config)
            pred_Y_epoch.data = np.ones(shape=pred_Y_epoch.data.shape)
            observed = pred_Y_epoch.multiply(Y_epoch)
            observed.eliminate_zeros()
            if Y_epoch.data.shape[0] > 0:
                LOGGER.info(
                    "Precision of values observed: {}".format(
                        observed.data.shape[0] / pred_Y_epoch.data.shape[0]
                    )
                )
            Y_epoch_top = smat_util.sorted_csr(Y_epoch, only_topk=self.eval_args["topk"])
            observed = smat.csr_matrix(observed, dtype=np.float32)
            epoch_rewards = np.array(np.sum(observed > 0, axis=1)).reshape(-1)
            rewards = np.concatenate([rewards, epoch_rewards])
            self.X_all = smat.csr_matrix(smat.vstack([self.X_all, X_epoch]))
            if self.inference_config["mode"] == "oracle":
                self.Ys_all = smat.csr_matrix(smat.vstack([self.Ys_all, Y_epoch_top]))
            else:
                self.Ys_all = smat.csr_matrix(smat.vstack([self.Ys_all, pred_Y_epoch]))
            self.Z_all += pred_maps
            self.Ya_all = smat.csr_matrix(smat.vstack([self.Ya_all, Y_epoch]))
        return rewards

    def _initialize_model(self):
        """Train first xmc  model using the initialization dataset."""
        LOGGER.info("Training initial model.")
        X = smat_util.sorted_csr(self.init_args["X"], only_topk=self.eval_args["topk"])
        Y = smat_util.sorted_csr(self.init_args["Y"], only_topk=self.eval_args["topk"])
        self.X_all = X
        self.Ya_all = Y.tocsr()
        self.Ys_all = Y.tocsr()
        self.Z_all = [[] for _ in range(X.shape[0])]
        init_config_train = copy.deepcopy(self.train_config)
        init_config_train["mode"] = "full"
        model = xcb_train.XFalconTrainer.train(
            self.X_all,
            self.Ys_all,
            self.Ya_all,
            self.Z_all,
            init_config_train,
            None,
        )
        LOGGER.info("Finished training and now saving init model.")
        model.save(self.model_path)
        gc.collect()

    def _get_inference_object(self):
        pred_config = copy.deepcopy(self.inference_config["pred_config"])
        if self.inference_config["mode"] in ["greedy", "oracle"]:
            inf_obj = xcb_inference.XLinearCBI.load(self.model_path)
            pred_config["multiplier"] = -1.0
        elif self.inference_config["mode"] == "falcon":
            inf_obj = xcb_inference.XLinearCBI.load(self.model_path)
            pred_config["multiplier"] *= self.X_all.shape[0]
        elif self.inference_config["mode"] == "e-greedy":
            inf_obj = xcb_inference.XLinearCBI.load(self.model_path)
        elif self.inference_config["mode"] == "boltzmann":
            pred_config["multiplier"] *= np.log(self.X_all.shape[0])
            inf_obj = xcb_inference.XLinearCBI.load(self.model_path)
        else:
            raise NotImplementedError("Inference mode not implemented")
        return inf_obj, pred_config

    def _create_schedule(self):
        """Create epoch schedule."""
        n = self.eval_args["X"].shape[0]
        if n != self.eval_args["Y"].shape[0]:
            raise ValueError("The shapes of eval X and Y do not match.")
        if self.eval_args["schedule"] == "linear":
            schedule = np.arange(0, n, self.eval_args["batch_size"])
            if schedule[-1] != n - 1:
                schedule = np.concatenate([schedule, [n]])
            else:
                schedule[-1] += 1
            return schedule
        elif self.eval_args["schedule"] == "exponential":
            schedule = [0]
            curr = 2 ** (int(np.log2(self.eval_args["batch_size"])))
            schedule.append(curr)
            while curr < n:
                curr = 2 * curr
                schedule.append(curr)
            schedule.append(n)
            return np.array(schedule)
        else:
            raise NotImplementedError("Other epoch schedules not implemented.")
