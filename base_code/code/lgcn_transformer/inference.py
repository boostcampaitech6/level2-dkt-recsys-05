from lgcn_transformer.args import parse_args
from lgcn_transformer.datasets import PrepareData
from lgcn_transformer import trainer
from lgcn_transformer.utils import get_logger, set_seeds, logging_conf, CFG


logger = get_logger(logging_conf)


def main(cfg):
    set_seeds(cfg.seed)

    cfg.train = False 

    logger.info("Preparing data ...")
    prepared = PrepareData(cfg).get_data()

    logger.info("Loading Model ...")
    model = trainer.build(prepared['merged_node'], cfg)

    logger.info("Make Predictions & Save Submission ...")
    trainer.inference(cfg, model=model, prepared=prepared, output_dir=cfg.output_dir)

if __name__ == "__main__":
    args = parse_args()
    cfg = CFG('config.yaml')

    for key, value in vars(args).items():
        if value is not None:  # 명령줄에서 제공된 인자만 업데이트
            setattr(cfg, key, value)

    main(cfg=cfg)

