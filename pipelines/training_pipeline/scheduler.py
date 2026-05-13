from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    ReduceLROnPlateau,
    OneCycleLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    PolynomialLR
)


class Scheduler:
    def __init__(self, optimizer, warmup, config, logger, tracker):
        self.config    = config
        self.optimizer = optimizer
        self.warmup    = warmup
        self.logger    = logger
        self.tracker   = tracker
        
        self.scheduler_type = getattr(self.config.scheduler, 'type', 'cosine_annealing')
        self.scheduler      = self._create_scheduler(self.scheduler_type)
        self._log_scheduler_info()
        
    def _create_scheduler(self, scheduler_type):
        scheduler_map = {
            'cosine_annealing'               : self._create_cosine_annealing,
            'step'                           : self._create_step,
            'multi_step'                     : self._create_multi_step,
            'exponential'                    : self._create_exponential,
            'reduce_on_plateau'              : self._create_reduce_on_plateau,
            'one_cycle'                      : self._create_one_cycle,
            'cosine_annealing_warm_restarts' : self._create_cosine_annealing_warm_restarts,
            'linear'                         : self._create_linear,
            'polynomial'                     : self._create_polynomial,
        }
        
        if scheduler_type not in scheduler_map:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        return scheduler_map[scheduler_type]()
    
    def _create_cosine_annealing(self):
        return CosineAnnealingLR(
            optimizer = self.optimizer,
            T_max     = self.config.scheduler.epochs,
            eta_min   = getattr(self.config.scheduler, 'eta_min', 0)
        )
    
    def _create_step(self):
        return StepLR(
            optimizer = self.optimizer,
            step_size = getattr(self.config.scheduler, 'step_size', 30),
            gamma     = getattr(self.config.scheduler, 'gamma', 0.1)
        )
    
    def _create_multi_step(self):
        return MultiStepLR(
            optimizer  = self.optimizer,
            milestones = getattr(self.config.scheduler, 'milestones', [30, 60, 90]),
            gamma      = getattr(self.config.scheduler, 'gamma', 0.1)
        )
    
    def _create_exponential(self):
        return ExponentialLR(
            optimizer = self.optimizer,
            gamma     = getattr(self.config.scheduler, 'gamma', 0.95)
        )
    
    def _create_reduce_on_plateau(self):
        return ReduceLROnPlateau(
            optimizer = self.optimizer,
            mode      = getattr(self.config.scheduler, 'mode', 'min'),
            factor    = getattr(self.config.scheduler, 'factor', 0.1),
            patience  = getattr(self.config.scheduler, 'patience', 10),
            threshold = getattr(self.config.scheduler, 'threshold', 1e-4),
            min_lr    = getattr(self.config.scheduler, 'min_lr', 0)
        )
    
    def _create_one_cycle(self):
        return OneCycleLR(
            optimizer       = self.optimizer,
            max_lr          = getattr(self.config.scheduler, 'max_lr', 0.1),
            total_steps     = getattr(self.config.scheduler, 'total_steps', 1000),
            pct_start       = getattr(self.config.scheduler, 'pct_start', 0.3),
            anneal_strategy = getattr(self.config.scheduler, 'anneal_strategy', 'cos')
        )
    
    def _create_cosine_annealing_warm_restarts(self):
        return CosineAnnealingWarmRestarts(
            optimizer = self.optimizer,
            T_0       = getattr(self.config.scheduler, 'T_0', 10),
            T_mult    = getattr(self.config.scheduler, 'T_mult', 2),
            eta_min   = getattr(self.config.scheduler, 'eta_min', 0)
        )
    
    def _create_linear(self):
        return LinearLR(
            optimizer    = self.optimizer,
            start_factor = getattr(self.config.scheduler, 'start_factor', 1.0),
            end_factor   = getattr(self.config.scheduler, 'end_factor', 0.1),
            total_iters  = getattr(self.config.scheduler, 'total_iters', 100)
        )
    
    def _create_polynomial(self):
        return PolynomialLR(
            optimizer    = self.optimizer,
            total_iters  = getattr(self.config.scheduler, 'total_iters', 100),
            power        = getattr(self.config.scheduler, 'power', 1.0)
        )
    
    def _log_scheduler_info(self):
        self.logger.section("[Learning Rate Scheduler]")
        self.logger.subsection(f"Scheduler Type    : {self.scheduler_type}")
        
        if self.scheduler_type == 'cosine_annealing':
            self.logger.subsection(f"T_max             : {self.config.scheduler.epochs}")
            self.logger.subsection(f"Eta Min           : {getattr(self.config.scheduler, 'eta_min', 0)}")
        
        elif self.scheduler_type == 'step':
            self.logger.subsection(f"Step Size         : {getattr(self.config.scheduler, 'step_size', 30)}")
            self.logger.subsection(f"Gamma             : {getattr(self.config.scheduler, 'gamma', 0.1)}")
        
        elif self.scheduler_type == 'multi_step':
            self.logger.subsection(f"Milestones        : {getattr(self.config.scheduler, 'milestones', [30, 60, 90])}")
            self.logger.subsection(f"Gamma             : {getattr(self.config.scheduler, 'gamma', 0.1)}")
        
        elif self.scheduler_type == 'exponential':
            self.logger.subsection(f"Gamma             : {getattr(self.config.scheduler, 'gamma', 0.95)}")
        
        elif self.scheduler_type == 'reduce_on_plateau':
            self.logger.subsection(f"Mode              : {getattr(self.config.scheduler, 'mode', 'min')}")
            self.logger.subsection(f"Factor            : {getattr(self.config.scheduler, 'factor', 0.1)}")
            self.logger.subsection(f"Patience          : {getattr(self.config.scheduler, 'patience', 10)}")
        
        elif self.scheduler_type == 'one_cycle':
            self.logger.subsection(f"Max LR            : {getattr(self.config.scheduler, 'max_lr', 0.1)}")
            self.logger.subsection(f"Total Steps       : {getattr(self.config.scheduler, 'total_steps', 1000)}")
        
        elif self.scheduler_type == 'cosine_annealing_warm_restarts':
            self.logger.subsection(f"T_0               : {getattr(self.config.scheduler, 'T_0', 10)}")
            self.logger.subsection(f"T_mult            : {getattr(self.config.scheduler, 'T_mult', 2)}")
        
        elif self.scheduler_type == 'linear':
            self.logger.subsection(f"Start Factor      : {getattr(self.config.scheduler, 'start_factor', 1.0)}")
            self.logger.subsection(f"End Factor        : {getattr(self.config.scheduler, 'end_factor', 0.1)}")
        
        elif self.scheduler_type == 'polynomial':
            self.logger.subsection(f"Total Iters       : {getattr(self.config.scheduler, 'total_iters', 100)}")
            self.logger.subsection(f"Power             : {getattr(self.config.scheduler, 'power', 1.0)}")
        
        self.logger.subsection(f"Warmup Enabled    : {self.warmup.enabled} \n")
        
    def step(self, epoch: int, metric=None) -> None:
        if self.warmup and not self.warmup.is_finished():
            return
        
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            group_name = param_group.get('name', f'group_{i}')
            self.tracker.log_scalar(f"lr/{group_name}", param_group['lr'], epoch)
    
    def state_dict(self) -> dict:
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict: dict) -> None:
        self.scheduler.load_state_dict(state_dict)
