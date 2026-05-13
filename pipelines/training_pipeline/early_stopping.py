import torch


class EarlyStopping:
    def __init__(self, config, logger, tracker):
        self.config           = config
        self.logger           = logger
        self.tracker          = tracker
        self.patience         = self.config.early_stopping.patience
        self.min_delta        = self.config.early_stopping.min_delta
        self.restore_best     = self.config.early_stopping.restore_best
        
        self.logger.section("[Early Stopping]")
        self.logger.subsection(f"Patience       : {self.patience}")
        self.logger.subsection(f"Min Delta      : {self.min_delta}")
        self.logger.subsection(f"Restore Best   : {self.restore_best} \n")

        self.best_loss        = None
        self.counter          = 0
        self.best_epoch       = -1
        self.best_model_state = None

    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None:
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self._save_state(model)
            self.counter    = 0
            stop            = False

        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self.counter    = 0
            self._save_state(model)
            self.tracker.log_scalar("early_stop/best_val_loss", self.best_loss, epoch)
            stop = False

        else:
            self.counter += 1
            stop = (self.counter >= self.patience)

        self.tracker.log_scalar("early_stop/counter", self.counter, epoch)

        if stop:
            self.logger.warning(f"Early stopping triggered at epoch {epoch + 1}. Restoring best model from epoch {self.best_epoch + 1}.")
            if self.restore_best:
                self.restore_model(model)

        return stop

    def _save_state(self, model):
        if self.restore_best:
            self.best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def restore_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state, strict=True)
