from asyncio.log import logger
import colossalai
from colossalai.logging import get_dist_logger, disable_existing_loggers

from vit_pytorch.vit import ViT
from vit_pytorch.extractor import Extractor

def ViT_Trainer():
    assert torch.cuda.is_available()
    disable_existing_loggers()

    colossalai.launch_from_torch(config='')

    vit = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    # setup dataloaders

    def build_dataloaders()

    train_dataloader, test_dataloader = build_dataloaders()

    # loss function

    loss_fn = nn.CrossEntropyLoss()

    # optimizer

    optimizer = optim.Adam(
        model.parameters(),
        lr = 0.0001
    )

    # intiailize ColossalAI

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize_engine(
        model=vit,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader
    )

    # Timer

    timer = MultiTimer()

    # Trainer

    trainer = Trainer(
        engine = engine,
        timer = timer,
        logger = logger
    )

    # Hooks

    hook_list = [
        hooks.LossHook(),
        hooks.AccuracyHook(accuracy_func = Accuracy()),
        hooks.LogMetricByEpochHook(logger),
    ]

    # Training Loop

    trainer.fit(
        
    )

