import numpy as np
import os
import ntpath
import time
from . import util3d as util
from . import html3d as html
from .animator3d import MedicalImageAnimator
from random import randrange


class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port = opt.display_port)
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            self.ncols = opt.display_ncols

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
    
    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch):
        if self.display_id > 0: # show images in the browser
        #if False:
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                </style>""" % (w, h)

                
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image_numpy in visuals.items():
                    #print("image numpy shape", label, image_numpy.shape)
                    if image_numpy.shape[2] == 1:
                        #print("Enter if", label)
                        label_html_row += '<td>%s</td>' % label
                        image = np.transpose(image_numpy, (2, 0, 1))
                        #print("image shape",image.shape)
                        images.append(image)
                    else:
                        image = np.expand_dims(image_numpy[127,:,:], axis=0)
                        image = (np.transpose(image, (1, 2, 0)) + 1) / 2.0 * 255.0
                        image = image.astype(np.uint8)
                        label_html_row += '<td>%s</td>' % label
                        #print("image shape",image.transpose([2, 0, 1]).shape)
                        images.append(image.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image.transpose([2, 0, 1]))*255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win = self.display_id + 2,
                              opts=dict(title=title + ' labels'))
            '''
            else:
                idx = 1
                for label, image_numpy in visuals.items():
                    #image_numpy = np.flipud(image_numpy)
                    self.vis.image(image_numpy.transpose([2,0,1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                    idx += 1
            '''
        '''
        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                if len(image_numpy.shape) == 4:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.gif' % (epoch, label))
                    animator = MedicalImageAnimator(image_numpy[0], [], 0, save=True)
                    animate = animator.run(img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                    util.save_image3d(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.gif' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()
        '''
    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)
    
    # gradients: last layer gradient for GAN and seg model
    def plot_current_grads(self, epoch, counter_ratio, opt, grads):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[],'Y':[], 'legend':list(grads.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([grads[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' grads over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'grads'},
            win=self.display_id)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
    
    # gradients
    def print_current_grads(self, epoch, i, grads):
        message = '(epoch: %d, iters: %d) ' % (epoch, i)
        for k, v in grads.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
    
    def print_current_metrics(self, epoch, MAE):
        message = '(epoch: %d, MAE: %.10f) ' % (epoch, MAE)
        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

# save image to the disk
def save_images(webpage, visuals, image_path, width=256):
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims = []
    txts = []
    links = []

    for label, image_numpy in visuals.items():
        image_name = '%s_%s_' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        #print("img numpy size", image_numpy.shape) #256x256x256
        util.save_image3d(image_numpy, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)

