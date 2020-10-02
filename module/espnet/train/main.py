import loadData as ld
import os
import torch
import pickle
import Model as net
from torch.autograd import Variable
import VisualizeGraph as viz
from Criteria import CrossEntropyLoss2d
import torch.backends.cudnn as cudnn
import Transforms as myTransforms
import DataSet as myDataLoader
import time
from argparse import ArgumentParser
from IOUEval import iouEval
import torch.optim.lr_scheduler
import cv2
import numpy as np

# modified https://github.com/sacmehta/ESPNet/blob/master/train/main.py by Issei Nakamura

pallete = [[0, 0, 0],
           [255, 0, 0],
           [0,255, 0],
           [255, 255, 0],
           [0,0,255],
           [128, 64, 128],
           [244, 35, 232],
           [70, 70, 70],
           [102, 102, 156],
           [190, 153, 153],
           [153, 153, 153],
           [250, 170, 30],
           [220, 220, 0],
           [107, 142, 35],
           [152, 251, 152],
           [70, 130, 180],
           [220, 20, 60],
           [255, 0, 0],
           [0, 0, 142],
           [0, 0, 70],
           [0, 60, 100],
           [0, 80, 100],
           [0, 0, 230],
           [119, 11, 32],
           [0, 0, 0]]


def val(args, val_loader, model, criterion, epoch):
    '''
    :param args: general arguments
    :param val_loader: loaded for validation dataset
    :param model: model
    :param criterion: loss function
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    #switch to evaluation mode
    model.eval()

    iouEvalVal = iouEval(args.classes)

    epoch_loss = []
    draw = 1

    total_batches = len(val_loader)
    for i, (input, target) in enumerate(val_loader):
        start_time = time.time()

        if type(args.gpu_id) is int:
            device = "cuda:{}".format(args.gpu_id)
            input = input.to(device)
            target = target.to(device)

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # run the mdoel
        output = model(input_var)

        # compute the loss
        loss = criterion(output, target_var)

        epoch_loss.append(loss.data)

        time_taken = time.time() - start_time

        # compute the confusion matrix
        iouEvalVal.addBatch(output.max(1)[1].data, target_var.data)

        print('[%d/%d] loss: %.3f time: %.2f' % (i, total_batches, loss.data, time_taken))
        if draw == 0:
            for ind in range(output.size()[0]):
                classMap_numpy = output[ind].max(0)[1].byte().cpu().data.numpy()
                input_y = 704
                input_x = 664
                classMap_numpy = cv2.resize(classMap_numpy,(input_y, input_x),interpolation=cv2.INTER_NEAREST)
                print("classMap_numpy shape:{}".format(classMap_numpy.shape))
                outdir = os.path.join(args.savedir, str(epoch))
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                classMap_numpy_color = np.zeros((input_x, input_y, 3), dtype=np.uint8)
                print("the shape of classMap_numpy_color :{}".format(classMap_numpy_color.shape))
                for idx in range(len(pallete)):
                    [r, g, b] = pallete[idx]
                    classMap_numpy_color[classMap_numpy == idx] = [b, g, r]
                cv2.imwrite(os.path.join(outdir, str(i*output.size()[0]+ind)+'.png'), classMap_numpy_color)

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)

    overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalVal.getMetric()

    return average_epoch_loss_val, overall_acc, per_class_acc, per_class_iu, mIOU

def train(args, train_loader, model, criterion, optimizer, epoch):
    '''
    :param args: general arguments
    :param train_loader: loaded for training dataset
    :param model: model
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    # switch to train mode
    model.train()

    iouEvalTrain = iouEval(args.classes)

    epoch_loss = []

    total_batches = len(train_loader)
    print("training")
    print("gpu id:{}".format(args.gpu_id))
    for i, (input, target) in enumerate(train_loader):
        start_time = time.time()

        if type(args.gpu_id) is int:
            device = "cuda:{}".format(args.gpu_id)
            input = input.to(device)
            target = target.to(device)
        else:
            print("CPU")

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        #run the mdoel
        output = model(input_var)

        #set the grad to zero
        optimizer.zero_grad()
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #epoch_loss.append(loss.data[0])
        epoch_loss.append(loss.data)
        time_taken = time.time() - start_time

        #compute the confusion matrix
        iouEvalTrain.addBatch(output.max(1)[1].data, target_var.data)

        print('[%d/%d] loss: %.3f time:%.2f' % (i, total_batches, loss.data, time_taken))

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalTrain.getMetric()

    return average_epoch_loss_train, overall_acc, per_class_acc, per_class_iu, mIOU

def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    '''
    helper function to save the checkpoint
    :param state: model state
    :param filenameCheckpoint: where to save the checkpoint
    :return: nothing
    '''
    torch.save(state, filenameCheckpoint)

def netParams(model):
    '''
    helper function to see total network parameters
    :param model: model
    :return: total network parameters
    '''
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters

def trainValidateSegmentation(args):
    '''
    Main function for trainign and validation
    :param args: global arguments
    :return: None
    '''
    # check if processed data file exists or not
    if not os.path.isfile(args.cached_data_file):
        dataLoad = ld.LoadData(args.data_dir, args.classes, args.cached_data_file)
        data = dataLoad.processData()
        if data is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        data = pickle.load(open(args.cached_data_file, "rb"))

    q = args.q
    p = args.p
    # load the model
    print("loading model")
    if not args.decoder:
        model = net.ESPNet_Encoder(args.classes, p=p, q=q)
        args.savedir = args.savedir + '_enc_' + str(p) + '_' + str(q) + '/'
    else:
        model = net.ESPNet(args.classes, p=p, q=q, encoderFile=args.pretrained)
        args.savedir = args.savedir + '_dec_' + str(p) + '_' + str(q) + '/'

    if type(args.gpu_id) is int:
        device = "cuda:{}".format(args.gpu_id)
        print("device:{}".format(device))
        model = model.to(device)
    else:
        print(type(args.gpu_id))
        print("CPU")

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    if args.visualizeNet:
        x = Variable(torch.randn(1, 3, args.inWidth, args.inHeight))

        if type(args.gpu_id) is int:
            x = x.to(device)

        y = model.forward(x)
        g = viz.make_dot(y)
        g.render(args.savedir + 'model.png', view=False)

    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    # define optimization criteria
    weight = torch.from_numpy(data['classWeights']) # convert the numpy array to torch
    if type(args.gpu_id) is int:
        device = "cuda:{}".format(args.gpu_id)
        weight = weight.to(device)

    criteria = CrossEntropyLoss2d(weight) #weight

    if type(args.gpu_id) is int:
        criteria = criteria.to(device)

    print('Data statistics')
    print(data['mean'], data['std'])
    print(data['classWeights'])
    output_stats_file = os.path.join(args.savedir, "mean_std.txt")
    with open(output_stats_file, "w") as out_f:
        out_f.write("mean[B G R]: {}\n".format(data['mean']))
        out_f.write("std[B G R]: {}".format(data['std']))


    #compose the data with transforms
    trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(1024, 512),
        myTransforms.RandomCropResize(32),
        myTransforms.RandomFlip(),
        #myTransforms.RandomCrop(64).
        myTransforms.ToTensor(args.scaleIn),
        #
    ])

    trainDataset_scale1 = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(1536, 768), # 1536, 768
        myTransforms.RandomCropResize(100),
        myTransforms.RandomFlip(),
        #myTransforms.RandomCrop(64),
        myTransforms.ToTensor(args.scaleIn),
        #
    ])

    trainDataset_scale2 = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(1280, 720), # 1536, 768
        myTransforms.RandomCropResize(100),
        myTransforms.RandomFlip(),
        #myTransforms.RandomCrop(64),
        myTransforms.ToTensor(args.scaleIn),
        #
    ])

    trainDataset_scale3 = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(768, 384),
        myTransforms.RandomCropResize(32),
        myTransforms.RandomFlip(),
        #myTransforms.RandomCrop(64),
        myTransforms.ToTensor(args.scaleIn),
        #
    ])

    trainDataset_scale4 = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(512, 256),
        #myTransforms.RandomCropResize(20),
        myTransforms.RandomFlip(),
        #myTransforms.RandomCrop(64).
        myTransforms.ToTensor(args.scaleIn),
        #
    ])


    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(1024, 512),
        myTransforms.ToTensor(args.scaleIn),
        #
    ])

    # since we training from scratch, we create data loaders at different scales
    # so that we can generate more augmented data and prevent the network from overfitting

    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_main),
        batch_size=args.batch_size + 2, shuffle=True, num_workers=args.num_workers, pin_memory=False)

    trainLoader_scale1 = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale1),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)

    trainLoader_scale2 = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale2),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)

    trainLoader_scale3 = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale3),
        batch_size=args.batch_size + 4, shuffle=True, num_workers=args.num_workers, pin_memory=False)

    trainLoader_scale4 = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale4),
        batch_size=args.batch_size + 4, shuffle=True, num_workers=args.num_workers, pin_memory=False)

    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['valIm'], data['valAnnot'], transform=valDataset),
        batch_size=args.batch_size + 4, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    if type(args.gpu_id) is int:
        cudnn.benchmark = True

    start_epoch = 0

    if args.resume:
        if os.path.isfile(args.resumeLoc):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resumeLoc)
            start_epoch = checkpoint['epoch']
            #args.lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_paramters)))
        logger.write("\n%s\t%s\t%s\t%s\t%s\t%s\t" % ('Epoch', 'Loss (train)', 'Loss (val)', 'mIoU (train)', 'mIoU (val)', 'Learning rate'))
    logger.flush()

    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    # we step the loss by 2 after step size is reached
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_loss, gamma=0.5)


    for epoch in range(start_epoch, args.max_epochs):

        scheduler.step(epoch)
        lr = 0
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " +  str(lr))

        # train for one epoch
        # We consider 1 epoch with all the training data (at different scales)
        print("scale1")
        train(args, trainLoader_scale1, model, criteria, optimizer, epoch)
        print("scale2")
        train(args, trainLoader_scale2, model, criteria, optimizer, epoch)
        print("scale4")
        train(args, trainLoader_scale4, model, criteria, optimizer, epoch)
        print("scale3")
        train(args, trainLoader_scale3, model, criteria, optimizer, epoch)
        print("scale main")
        lossTr, overall_acc_tr, per_class_acc_tr, per_class_iu_tr, mIOU_tr = train(args, trainLoader, model, criteria, optimizer, epoch)

        # evaluate on validation set
        print("validation")
        lossVal, overall_acc_val, per_class_acc_val, per_class_iu_val, mIOU_val = val(args, valLoader, model, criteria, epoch)
        
            
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lossTr': lossTr,
            'lossVal': lossVal,
            'iouTr': mIOU_tr,
            'iouVal': mIOU_val,
            'lr': lr
        }, args.savedir + 'checkpoint.pth.tar')

        #save the model also
        model_file_name = args.savedir + '/model_' + str(epoch + 1) + '.pth'
        torch.save(model.state_dict(), model_file_name)

        

        with open(args.savedir + 'acc_' + str(epoch) + '.txt', 'w') as log:
            log.write("\nEpoch: %d\t Overall Acc (Tr): %.4f\t Overall Acc (Val): %.4f\t mIOU (Tr): %.4f\t mIOU (Val): %.4f" % (epoch, overall_acc_tr, overall_acc_val, mIOU_tr, mIOU_val))
            log.write('\n')
            log.write('Per Class Training Acc: ' + str(per_class_acc_tr))
            log.write('\n')
            log.write('Per Class Validation Acc: ' + str(per_class_acc_val))
            log.write('\n')
            log.write('Per Class Training mIOU: ' + str(per_class_iu_tr))
            log.write('\n')
            log.write('Per Class Validation mIOU: ' + str(per_class_iu_val))

        logger.write("\n%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.7f" % (epoch, lossTr, lossVal, mIOU_tr, mIOU_val, lr))
        logger.flush()
        print("Epoch : " + str(epoch) + ' Details')
        print("\nEpoch No.: %d\tTrain Loss = %.4f\tVal Loss = %.4f\t mIOU(tr) = %.4f\t mIOU(val) = %.4f" % (epoch, lossTr, lossVal, mIOU_tr, mIOU_val))
    logger.close()



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model', default="ESPNet", help='Set model name')
    parser.add_argument('--data_dir', default="./city", help='Set data directory')
    parser.add_argument('--inWidth', type=int, default=1024, help='Set width of RGB image')
    parser.add_argument('--inHeight', type=int, default=512, help='Set height of RGB image')
    parser.add_argument('--scaleIn', type=int, default=8, help='For ESPNet-C, scaleIn=8. For ESPNet, scaleIn=1')
    parser.add_argument('--max_epochs', type=int, default=300, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. '
                                                                   'Change as per the GPU memory')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--savedir', default='./results_enc_', help='Set path to directory to save the results')
    parser.add_argument('--visualizeNet', type=bool, default=True, help='If you want to visualize the model structure')
    parser.add_argument('--resume', type=bool, default=False, help='Use this flag to load last checkpoint for training')  #
    parser.add_argument('--classes', type=int, default=20, help='No of classes in the dataset. 20 for cityscapes')
    parser.add_argument('--cached_data_file', default='city.p', help='Cached file name')
    parser.add_argument('--logFile', default='trainValLog.txt', help='Set path to a file that stores the training and validation logs')
    parser.add_argument('--gpu_id', default=0, type=int, help='Set gpu id. If -1, then CPU.')
    parser.add_argument('--decoder', type=bool, default=False,help='True if ESPNet. False for ESPNet-C') # False for encoder
    parser.add_argument('--pretrained', default='../pretrained/encoder/espnet_p_2_q_8.pth', help='Pretrained ESPNet-C weights. '
                                                                              'Only used when training ESPNet')
    parser.add_argument('--p', default=2, type=int, help='depth multiplier')
    parser.add_argument('--q', default=8, type=int, help='depth multiplier')

    trainValidateSegmentation(parser.parse_args())

