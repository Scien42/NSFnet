# Copyright (c) 2023 scien42.tech, Se42 Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Author: Zhicheng Wang, Hui Xiang
# Created: 08.03.2023
import os
import torch
import scipy.io
import numpy as np
from net import FCNet
from typing import Dict, List, Set, Optional, Union, Callable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PysicsInformedNeuralNetwork:
    # Initialize the class
    def __init__(self,
                 opt=None,
                 Re = 1000,
                 layers=6,
                 layers_1=6,
                 hidden_size=80,
                 hidden_size_1=20,
                 N_f = 100000,
                 alpha_evm=0.03,
                 learning_rate=0.001,
                 weight_decay=0.9,
                 outlet_weight=1,
                 bc_weight=1,
                 eq_weight=1,
                 ic_weight=1,
                 num_ins=2,
                 num_outs=3,
                 num_outs_1=1,
                 supervised_data_weight=1,
                 net_params=None,
                 net_params_1=None,
                 checkpoint_freq=2000,
                 checkpoint_path='./checkpoint/'):

        self.evm = None
        self.Re = Re
        self.vis_t0 = 20.0/self.Re

        self.layers = layers
        self.layers_1 = layers_1
        self.hidden_size = hidden_size
        self.hidden_size_1 = hidden_size_1
        self.N_f = N_f

        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_path = checkpoint_path

        self.alpha_evm = alpha_evm
        self.alpha_b = bc_weight
        self.alpha_e = eq_weight
        self.alpha_i = ic_weight
        self.alpha_o = outlet_weight
        self.alpha_s = supervised_data_weight
        self.loss_i = self.loss_o = self.loss_b = self.loss_e = self.loss_s = 0.0

        # initialize NN
        self.net = self.initialize_NN(
                num_ins=num_ins, num_outs=num_outs, num_layers=layers, hidden_size=hidden_size).to(device)
        self.net_1 = self.initialize_NN(
                num_ins=num_ins, num_outs=num_outs_1, num_layers=layers_1, hidden_size=hidden_size_1).to(device)
        if net_params:
            load_params = torch.load(net_params)
            self.net.load_state_dict(load_params)

        if net_params_1:
            load_params_1 = torch.load(net_params_1)
            self.net_1.load_state_dict(load_params_1)

        self.opt = torch.optim.Adam(
            list(self.net.parameters())+list(self.net_1.parameters()),
            lr=learning_rate,
            weight_decay=0.0) if not opt else opt

    def init_vis_t(self):
        (_,_,_,e) = self.neural_net_u(self.x_f, self.y_f)
        self.vis_t_minus  = self.alpha_evm*torch.abs(e).detach().cpu().numpy()

    def set_boundary_data(self, X=None, time=False):
        # boundary training data | u, v, t, x, y
        requires_grad = False
        self.x_b = torch.tensor(X[0], requires_grad=requires_grad).float().to(device)
        self.y_b = torch.tensor(X[1], requires_grad=requires_grad).float().to(device)
        self.u_b = torch.tensor(X[2], requires_grad=requires_grad).float().to(device)
        self.v_b = torch.tensor(X[3], requires_grad=requires_grad).float().to(device)
        if time:
            self.t_b = torch.tensor(X[4], requires_grad=requires_grad).float().to(device)

    def set_eq_training_data(self,
                             X=None,
                             time=False):
        requires_grad = True
        self.x_f = torch.tensor(X[0], requires_grad=requires_grad).float().to(device)
        self.y_f = torch.tensor(X[1], requires_grad=requires_grad).float().to(device)
        if time:
            self.t_f = torch.tensor(X[2], requires_grad=requires_grad).float().to(device)

        self.init_vis_t()

    def set_optimizers(self, opt):
        self.opt = opt

    def set_alpha_evm(self, alpha):
        self.alpha_evm = alpha

    def initialize_NN(self,
                      num_ins=3,
                      num_outs=3,
                      num_layers=10,
                      hidden_size=50):
        return FCNet(num_ins=num_ins,
                     num_outs=num_outs,
                     num_layers=num_layers,
                     hidden_size=hidden_size,
                     activation=torch.nn.Tanh)

    def set_eq_training_func(self, train_data_func):
        self.train_data_func = train_data_func

    def neural_net_u(self, x, y):
        X = torch.cat((x, y), dim=1)
        uvp = self.net(X)
        ee = self.net_1(X)
        u = uvp[:, 0]
        v = uvp[:, 1]
        p = uvp[:, 2:3]
        e =  ee[:,0:1]
        return u, v, p, e

    def neural_net_equations(self, x, y):
        X = torch.cat((x, y), dim=1)
        uvp = self.net(X)
        ee = self.net_1(X)

        u = uvp[:, 0:1]
        v = uvp[:, 1:2]
        p = uvp[:, 2:3]
        e = ee[:, 0:1]
        self.evm = e

        u_x, u_y = self.autograd(u, [x,y])
        u_xx = self.autograd(u_x, [x])[0]
        u_yy = self.autograd(u_y, [y])[0]

        v_x, v_y = self.autograd(v, [x,y])
        v_xx = self.autograd(v_x, [x])[0]
        v_yy = self.autograd(v_y, [y])[0]

        p_x, p_y = self.autograd(p, [x,y])

        # Get the minum between (vis_t0, vis_t_mius(calculated with last step e))
        self.vis_t = torch.tensor(
                np.minimum(self.vis_t0, self.vis_t_minus)).float().to(device)
        # Save vis_t_minus for computing vis_t in the next step
        self.vis_t_minus  = self.alpha_evm*torch.abs(e).detach().cpu().numpy()

        # NS
        eq1 = (u*u_x + v*u_y) + p_x - (1.0/self.Re+self.vis_t)*(u_xx + u_yy)
        eq2 = (u*v_x + v*v_y) + p_y - (1.0/self.Re+self.vis_t)*(v_xx + v_yy)
#        eq11 = (u*u_x + v*u_y) + p_x - 1.0/self.Re*(u_xx + u_yy)
#        eq22 = (u*v_x + v*v_y) + p_y - 1.0/self.Re*(v_xx + v_yy)
        eq3 = u_x + v_y

        residual = (eq1*(u-0.5)+eq2*(v-0.5))-e
        return eq1, eq2, eq3, residual

    @torch.jit.script
    def autograd(y: torch.Tensor, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        TorchScript function to compute the gradient of a tensor wrt multople inputs
        """
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y, device=y.device)]
        grad = torch.autograd.grad(
            [
                y,
            ],
            x,
            grad_outputs=grad_outputs,
            create_graph=True,
            allow_unused=True,
            #retain_graph=True,
        )

        if grad is None:
            grad = [torch.zeros_like(xx) for xx in x]
        assert grad is not None
        grad = [g if g is not None else torch.zeros_like(x[i]) for i, g in enumerate(grad)]
        return grad

    def predict(self, net_params, X):
        x, y = X
        return self.neural_net_u(x, y)

    def shuffle(self, tensor):
        tensor_to_numpy = tensor.detach().cpu()
        shuffle_numpy = np.random.shuffle(tensor_to_numpy)
        return torch.tensor(tensor_to_numpy, requires_grad=True).float()

    def fwd_computing_loss_2d(self, loss_mode='MSE'):
        # boundary data
        (self.u_pred_b, self.v_pred_b, _, _) = self.neural_net_u(self.x_b, self.y_b)

        # BC loss
        if loss_mode == 'L2':
            self.loss_b = torch.norm((self.u_b.reshape([-1]) - self.u_pred_b.reshape([-1])), p=2) + \
                          torch.norm((self.v_b.reshape([-1]) - self.v_pred_b.reshape([-1])), p=2)
        if loss_mode == 'MSE':
            self.loss_b = torch.mean(torch.square(self.u_b.reshape([-1]) - self.u_pred_b.reshape([-1]))) + \
                          torch.mean(torch.square(self.v_b.reshape([-1]) - self.v_pred_b.reshape([-1])))

        # equation
        assert self.x_f is not None and self.y_f is not None

        (self.eq1_pred, self.eq2_pred,
         self.eq3_pred, self.eq4_pred) = self.neural_net_equations(self.x_f, self.y_f)
        if loss_mode == 'L2':
            self.loss_e = torch.norm(self.eq1_pred.reshape([-1]), p=2) + \
                          torch.norm(self.eq2_pred.reshape([-1]), p=2) + \
                          torch.norm(self.eq3_pred.reshape([-1]), p=2)
        if loss_mode == 'MSE':
            self.loss_eq1 = torch.mean(torch.square(self.eq1_pred.reshape([-1])))
            self.loss_eq2 = torch.mean(torch.square(self.eq2_pred.reshape([-1])))
            self.loss_eq3 = torch.mean(torch.square(self.eq3_pred.reshape([-1])))
            self.loss_eq4 = torch.mean(torch.square(self.eq4_pred.reshape([-1])))
            self.loss_e = self.loss_eq1+self.loss_eq2+self.loss_eq3 + 0.1*self.loss_eq4

        self.loss = self.alpha_b * self.loss_b + self.alpha_e * self.loss_e

        return self.loss, [self.loss_e, self.loss_b]

    def train(self,
              num_epoch=1,
              lr=1e-4,
              optimizer=None,
              scheduler=None,
              batchsize=None):
        if self.opt is not None:
            self.opt.param_groups[0]['lr'] = lr
        return self.solve_Adam(self.fwd_computing_loss_2d, num_epoch, batchsize, scheduler)

    def solve_Adam(self,
                   loss_func,
                   num_epoch=1000,
                   batchsize=None,
                   scheduler=None):
        self.freeze_evm_net(0)
        for epoch_id in range(num_epoch):
            # train evm net every 10000 step
            if epoch_id !=0 and epoch_id % 10000 == 0:
                self.defreeze_evm_net(epoch_id)
            if (epoch_id - 1) % 10000 == 0:
                self.freeze_evm_net(epoch_id)

            loss, losses = loss_func()
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            if scheduler:
                scheduler.step()

            if epoch_id == 0 or (epoch_id + 1)%100 == 0:
                self.print_log(loss, losses, epoch_id, num_epoch)

            if epoch_id == 0 or (epoch_id)%10000 == 0:
                saved_ckpt = 'model_cavity_loop%d.pth'%(epoch_id)
                layers = self.layers
                hidden_size = self.hidden_size
                N_f = self.N_f
                self.save(saved_ckpt, N_HLayer=layers, N_neu=hidden_size, N_f=N_f)

            '''
            if epoch_id % 1000 == 0:
                print(epoch_id)
                print(self.evm)
            '''

    def freeze_evm_net(self, epoch_id):
       # print("*"*20)
       # print("freeze evm net")
        #print(epoch_id)
        #if self.evm is not None:
        #    print(self.evm)
        for para in self.net_1.parameters():
            para.requires_grad = False
        self.opt.param_groups[0]['params'] = list(self.net.parameters())
       # print("*"*20)

    def defreeze_evm_net(self, epoch_id):
       # print("*"*20)
       # print("defreeze evm net")
        #print(epoch_id)
        #print(self.evm)
        for para in self.net_1.parameters():
            para.requires_grad = True
        self.opt.param_groups[0]['params'] = list(self.net.parameters())+list(self.net_1.parameters())
       # print("*"*20)

    def print_log(self, loss, losses, epoch_id, num_epoch):
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        print("current lr is {}".format(get_lr(self.opt)))
        if isinstance(losses[0], int):
            eq_loss = losses[0]
        else:
            eq_loss = losses[0].detach().cpu().item()

        print("epoch/num_epoch: ", epoch_id + 1, "/", num_epoch,
              "loss[Adam]: %.3e"
              %(loss.detach().cpu().item()), 
              "eq1_loss: %.3e " %(self.loss_eq1.detach().cpu().item()),
              "eq2_loss: %.3e " %(self.loss_eq2.detach().cpu().item()),
              "eq3_loss: %.3e " %(self.loss_eq3.detach().cpu().item()),
              "eq4_loss: %.3e " %(self.loss_eq4.detach().cpu().item()),
              "bc_loss: %.3e" %(losses[1].detach().cpu().item())) 

        '''
        if (epoch_id + 1) % self.checkpoint_freq == 0:
            torch.save(
                self.net.state_dict(),
                self.checkpoint_path + 'net_params_' + str(epoch_id + 1) + '.pth')
        '''

    def evaluate(self, x, y, u, v):
        """ testing all points in the domain """
        x_test = x.reshape(-1,1)
        y_test = y.reshape(-1,1)
        u_test = u.reshape(-1,1)
        v_test = v.reshape(-1,1)

        x_test = torch.tensor(x_test).float().to(device)
        y_test = torch.tensor(y_test).float().to(device)
        u_pred, v_pred, _, _= self.neural_net_u(x_test, y_test)
        u_pred = u_pred.detach().cpu().numpy().reshape(-1,1)
        v_pred = v_pred.detach().cpu().numpy().reshape(-1,1)
        # Error
        error_u = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
        error_v = np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2)
        print('------------------------')
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))

    def test(self, x, y, u, v,loop=None):
        """ testing all points in the domain """
        x_test = x.reshape(-1,1)
        y_test = y.reshape(-1,1)
        u_test = u.reshape(-1,1)
        v_test = v.reshape(-1,1)
        # Prediction
        x_test = torch.tensor(x_test).float().to(device)
        y_test = torch.tensor(y_test).float().to(device)
        u_pred, v_pred, p_pred, e_pred= self.neural_net_u(x_test, y_test)
        u_pred = u_pred.detach().cpu().numpy().reshape(-1,1)
        v_pred = v_pred.detach().cpu().numpy().reshape(-1,1)
        p_pred = p_pred.detach().cpu().numpy().reshape(-1,1)
        e_pred = e_pred.detach().cpu().numpy().reshape(-1,1)
        # Error
        error_u = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
        error_v = np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2)
        print('------------------------')
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('------------------------')

#        div_pred = self.divergence(x_test, y_test)
        u_pred = u_pred.reshape(257,257)
        v_pred = v_pred.reshape(257,257)
        p_pred = p_pred.reshape(257,257)
        e_pred = e_pred.reshape(257,257)
#        div_pred = div_pred.detach().cpu().numpy().reshape(257,257)

        scipy.io.savemat('cavity_result_loop_%d.mat'%(loop),
                    {'U_pred':u_pred,
                     'V_pred':v_pred,
                     'P_pred':p_pred,
                     'E_pred':e_pred,
                     'lam_bcs':self.alpha_b,
                     'lam_equ':self.alpha_e})

    def save(self, filename, directory=None, N_HLayer=None, N_neu=None, N_f=None):
        Re_folder = 'Re'+str(self.Re)
        NNsize = str(N_HLayer) + 'x' + str(N_neu) + '_Nf'+str(np.int32(N_f/1000)) + 'k'
        lambdas = 'lamB'+str(self.alpha_b) + '_alpha'+str(self.alpha_evm)

        relative_path = '/results/' +  Re_folder + '/' + NNsize + '_' + lambdas + '/'

        if not directory:
            directory = os.getcwd()
        save_results_to = directory + relative_path
        if not os.path.exists(save_results_to):
            os.makedirs(save_results_to)
        
        torch.save(self.net.state_dict(), save_results_to+filename)
        torch.save(self.net_1.state_dict(), save_results_to+filename+'_evm')


    def divergence(self, x_star, y_star):
        (self.eq1_pred, self.eq2_pred,
         self.eq3_pred, self.eq4_pred) = self.neural_net_equations(self.x_star, self.y_star)
        div = self.eq3_pred
        return div
