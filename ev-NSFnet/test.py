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
import torch
from tools import *
import cavity_data as cavity
import pinn_solver as psolver


def train(net_params=None, net_params_1=None, loop = 0):
    Re = 5000   # Reynolds number
    N_neu = 120
    N_neu_1 = 40
    lam_bcs = 10
    lam_equ = 1
    N_f = 200000
    alpha_evm = 0.03
    N_HLayer = 4
    N_HLayer_1 = 4
  #  layers = [2] + N_HLayer*[N_neu] + [4]

    PINN = psolver.PysicsInformedNeuralNetwork(
        Re=Re,
        layers=N_HLayer,
        layers_1=N_HLayer_1,
        hidden_size = N_neu,
        hidden_size_1 = N_neu_1,
        alpha_evm=alpha_evm,
        bc_weight=lam_bcs,
        eq_weight=lam_equ,
        net_params=net_params,
        net_params_1=net_params_1,
        checkpoint_path='./checkpoint/')

    path = './datasets/'
    dataloader = cavity.DataLoader(path=path, N_f=N_f, N_b=1000)

    filename = './data/cavity_Re'+str(Re)+'_256_Uniform.mat'
    x_star, y_star, u_star, v_star = dataloader.loading_evaluate_data(filename)

    # Evaluating
    PINN.evaluate(x_star, y_star, u_star, v_star)
    PINN.test(x_star, y_star, u_star, v_star, loop)

if __name__ == "__main__":
    
    for eid in range(0, 500000, 10000):
       net_params = 'results/Re5000/4x120_Nf200k_lamB10_alpha0.05/model_cavity_loop%d.pth'%(eid)
       net_params_1 = 'results/Re5000/4x120_Nf200k_lamB10_alpha0.05/model_cavity_loop%d.pth_evm'%(eid)
       train(net_params=net_params, net_params_1 = net_params_1, loop = eid)

    for eid in range(0, 500000, 10000):
       net_params = 'results/Re5000/4x120_Nf200k_lamB10_alpha0.03/model_cavity_loop%d.pth'%(eid)
       net_params_1 = 'results/Re5000/4x120_Nf200k_lamB10_alpha0.03/model_cavity_loop%d.pth_evm'%(eid)
       train(net_params=net_params, net_params_1 = net_params_1, loop = eid+500000)
       
    for eid in range(0, 500000, 10000):
       net_params = 'results/Re5000/4x120_Nf200k_lamB10_alpha0.02/model_cavity_loop%d.pth'%(eid)
       net_params_1 = 'results/Re5000/4x120_Nf200k_lamB10_alpha0.02/model_cavity_loop%d.pth_evm'%(eid)
       train(net_params=net_params, net_params_1 = net_params_1, loop = eid+1000000)

    for eid in range(0, 500000, 10000):
       net_params = 'results/Re5000/4x120_Nf200k_lamB10_alpha0.01/model_cavity_loop%d.pth'%(eid)
       net_params_1 = 'results/Re5000/4x120_Nf200k_lamB10_alpha0.01/model_cavity_loop%d.pth_evm'%(eid)
       train(net_params=net_params, net_params_1 = net_params_1, loop = eid+1500000)

    for eid in range(0, 500000, 10000):
       net_params = 'results/Re5000/4x120_Nf200k_lamB10_alpha0.005/model_cavity_loop%d.pth'%(eid)
       net_params_1 = 'results/Re5000/4x120_Nf200k_lamB10_alpha0.005/model_cavity_loop%d.pth_evm'%(eid)
       train(net_params=net_params, net_params_1 = net_params_1, loop = eid+2000000)

    for eid in range(0, 500000, 10000):
       net_params = 'results/Re5000/4x120_Nf200k_lamB10_alpha0.002/model_cavity_loop%d.pth'%(eid)
       net_params_1 = 'results/Re5000/4x120_Nf200k_lamB10_alpha0.002/model_cavity_loop%d.pth_evm'%(eid)
       train(net_params=net_params, net_params_1 = net_params_1, loop = eid+2500000)
