import torch

from torchvision.datasets import CIFAR10, ImageNet
from torchvision import transforms as T

import colossalai
from colossalai.logging import get_dist_logger, disable_existing_loggers
from colossalai.core import global_context as gpc
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_dataloader
from colossalai.nn.metric import Accuracy

from vit_pytorch.vit import ViT
from vit_pytorch.extractor import Extractor

def ViT_Trainer():
    assert torch.cuda.is_available()
    disable_existing_loggers()

    colossalai.launch_from_torch(config='./vit_config')

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

    train_transform = T.Compose([
            T.Resize(args.image_size),
            T.AutoAugment(policy = policy),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transform = T.Compose([
            T.Resize(args.image_size),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = ImageNet(
        root = args.path_to_data,
        train = True,
        download = args.download_dataset,
        transform = train_transform,
    )
        
    test_dataset = ImageNet(
        root = args.path_to_data,
        train = False,
        download = args.download_dataset,
        transform = test_transform,
    )    

    train_loader = get_dataloader(
        train_dataset, 
        shuffle = args.shuffle,
        batch_size = args.batch_size, 
        seed = args.seed, 
        add_sampler = args.add_sampler, 
        drop_last = args.drop_last, 
        pin_memory = args.pin_memory, 
        num_workers = args.num_workers
    )
    
    test_loader = get_dataloader(
        test_dataset, 
        batch_size = args.batch_size, 
        seed = args.seed, 
        add_sampler = False, 
        drop_last = args.drop_last, 
        pin_memory = args.pin_memory, 
        num_workers = args.num_workers
    )

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
        train_dataloader = train_dataloader,
        epochs = epochs,
        test_dataloader = test_dataloader,
        hook_list = hook_list,
        display_progress = True,
        test_interval = test_interval
    )

