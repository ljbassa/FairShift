import itertools
import numpy as np
import os
import torch

class BaseTrainer:
    def __init__(self,
                 hyper_space,
                 search_priority_increasing,
                 patience):
        """Base class for training a discriminative model.

        Parameters
        ----------
        search_priority_increasing : list of str
            The priority of hyperparameters to search, from lowest to highest.
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.hyper_space = hyper_space
        self.search_priority_increasing = search_priority_increasing
        self.patience = patience

    def get_config_list(self):
        vals = [self.hyper_space[k] for k in self.search_priority_increasing]

        config_list = []
        for items in itertools.product(*vals):
            items_dict = dict(zip(self.search_priority_increasing, items))
            config_list.append(items_dict)

        return config_list

    def save_model(self, model_path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": self.best_model_config
        }, model_path)

class BaseEvaluator:
    def __init__(self,
                 Trainer,
                 model_path,
                 num_classes,
                 train_mask,
                 val_mask,
                 test_mask,
                 **real_data):
        self.Trainer = Trainer
        self.num_classes = num_classes
        self.train_mask_real = train_mask
        self.val_mask_real = val_mask
        self.test_mask_real = test_mask
        self.real_data = real_data

        self.model_real = Trainer()
        #if os.path.exists(model_path):
        #    self.model_real.load_model(model_path)
        #else:
        self.model_real.fit(num_classes=num_classes,
                            train_mask=train_mask,
                            val_mask=val_mask,
                            **real_data)
        #self.model_real.save_model(model_path)

        self.real_real_acc, self.real_real_sp, self.real_real_eo = self.model_real.predict(
            mask=test_mask, **real_data)

        self.real_sample_acc = []
        self.sample_real_acc = []
        self.sample_sample_acc = []
        
        self.real_sample_sp = []
        self.sample_real_sp = []
        self.sample_sample_sp = []
        
        self.real_sample_eo = []
        self.sample_real_eo = []
        self.sample_sample_eo = []

    def add_sample(self,
                   train_mask,
                   val_mask,
                   test_mask,
                   **sample_data):
        
        real_sample_acc, real_sample_sp, real_sample_eo = self.model_real.predict(mask=test_mask, **sample_data)
        self.real_sample_acc.append(real_sample_acc)
        self.real_sample_sp.append(real_sample_sp)
        self.real_sample_eo.append(real_sample_eo)

        model_sample = self.Trainer()
        model_sample.fit(num_classes=self.num_classes,
                         train_mask=train_mask,
                         val_mask=val_mask,
                         **sample_data)
        
        sample_real_acc, sample_real_sp, sample_real_eo = model_sample.predict(mask=self.test_mask_real, **self.real_data)

        self.sample_real_acc.append(sample_real_acc)
        self.sample_real_sp.append(sample_real_sp)
        self.sample_real_eo.append(sample_real_eo)
        
        sample_sample_acc, sample_sample_sp, sample_sample_eo = model_sample.predict(mask=test_mask, **sample_data)

        self.sample_sample_acc.append(sample_sample_acc)
        self.sample_sample_sp.append(sample_sample_sp)
        self.sample_sample_eo.append(sample_sample_eo)

    def summary(self):
        print(f"ACC/AUC(G|G): {self.real_real_acc}")
        print(f"SP(G|G): {self.real_real_sp}")
        print(f"EO(G|G): {self.real_real_eo}")
        
        mean_sample_real_acc = np.mean(self.sample_real_acc)
        mean_sample_real_sp = np.mean(self.sample_real_sp)
        mean_sample_real_eo = np.mean(self.sample_real_eo)
        std_sample_real_acc = np.std(self.sample_real_acc)
        std_sample_real_sp = np.std(self.sample_real_sp)
        std_sample_real_eo = np.std(self.sample_real_eo)
        print(f"ACC/AUC(G|G_hat): {mean_sample_real_acc}")
        print(f"SP(G|G_hat): {mean_sample_real_sp}")
        print(f"EO(G|G_hat): {mean_sample_real_eo}")
        print(f"STD of ACC/AUC(G|G_hat): {std_sample_real_acc}")
        print(f"STD of SP(G|G_hat): {std_sample_real_sp}")
        print(f"STD of EO(G|G_hat): {std_sample_real_eo}")

        mean_sample_sample_acc = np.mean(self.sample_sample_acc)
        mean_sample_sample_sp = np.mean(self.sample_sample_sp)
        mean_sample_sample_eo = np.mean(self.sample_sample_eo)
        print(f"ACC/AUC(G_hat|G_hat): {mean_sample_sample_acc}")
        print(f"SP(G_hat|G_hat): {mean_sample_sample_sp}")
        print(f"EO(G_hat|G_hat): {mean_sample_sample_eo}")
        
        mean_real_sample_acc = np.mean(self.real_sample_acc)
        mean_real_sample_sp = np.mean(self.real_sample_sp)
        mean_real_sample_eo = np.mean(self.real_sample_eo)
        print(f"ACC/AUC(G_hat|G): {mean_real_sample_acc}")
        print(f"SP(G_hat|G): {mean_real_sample_sp}")
        print(f"EO(G_hat|G): {mean_real_sample_eo}")
