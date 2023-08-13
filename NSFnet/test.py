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


def train(net_params=None,loop = 0):
    Re = 2000   # Reynolds number
    N_neu = 120
    N_neu_1 = 40
    lam_bcs = 10
    lam_equ = 1
    N_f = 40000
    N_HLayer = 4
  #  layers = [2] + N_HLayer*[N_neu] + [4]

    PINN = psolver.PysicsInformedNeuralNetwork(
        Re=Re,
        layers=N_HLayer,
        hidden_size = N_neu,
        N_f = N_f,
        bc_weight=lam_bcs,
        eq_weight=lam_equ,
        num_ins = 2,
        num_outs = 3,
        net_params=net_params,
        checkpoint_path='./checkpoint/')

    path = './datasets/'
    dataloader = cavity.DataLoader(path=path, N_f=N_f, N_b=1000)

    filename = './data/cavity_Re'+str(Re)+'_256.mat'
    x_star, y_star, u_star, v_star = dataloader.loading_evaluate_data(filename)

    # Evaluating
    PINN.evaluate(x_star, y_star, u_star, v_star)
    PINN.test(x_star, y_star, u_star, v_star, loop)

if __name__ == "__main__":
    
    for eid in range(10000, 210000, 10000):
       net_params = 'results/Re2000/4x120_Nf40k_lamB10_alpha0.031/model_cavity_loop_%d.pth'%(eid)
       train(net_params=net_params, loop = eid)

    for eid in range(10000, 210000, 10000):
       net_params = 'results/Re2000/4x120_Nf40k_lamB10_alpha0.032/model_cavity_loop_%d.pth'%(eid)
       train(net_params=net_params, loop = eid+200000)
       
    for eid in range(10000, 210000, 10000):
       net_params = 'results/Re2000/4x120_Nf40k_lamB10_alpha0.033/model_cavity_loop_%d.pth'%(eid)
       train(net_params=net_params, loop = eid+400000)

    for eid in range(10000, 510000, 10000):
       net_params = 'results/Re2000/4x120_Nf40k_lamB10_alpha0.034/model_cavity_loop_%d.pth'%(eid)
       train(net_params=net_params, loop = eid+600000)

    for eid in range(10000, 510000, 10000):
       net_params = 'results/Re2000/4x120_Nf40k_lamB10_alpha0.035/model_cavity_loop_%d.pth'%(eid)
       train(net_params=net_params, loop = eid+1100000)

