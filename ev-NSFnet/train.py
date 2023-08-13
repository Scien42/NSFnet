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


def train(net_params=None):
    Re = 5000   # Reynolds number
    N_neu = 120
    N_neu_1 = 40
    lam_bcs = 10
    lam_equ = 1
    N_f = 120000
    #N_f = 120000
    alpha_evm = 0.03
    N_HLayer = 4
    N_HLayer_1 = 4
#    layers = [2] + N_HLayer*[N_neu] + [4]

    PINN = psolver.PysicsInformedNeuralNetwork(
        Re=Re,
        layers=N_HLayer,
        layers_1=N_HLayer_1,
        hidden_size = N_neu,
        hidden_size_1 = N_neu_1,
        N_f = N_f,
        alpha_evm=alpha_evm,
        bc_weight=lam_bcs,
        eq_weight=lam_equ,
        net_params=net_params,
        checkpoint_path='./checkpoint/')

    path = './datasets/'
    dataloader = cavity.DataLoader(path=path, N_f=N_f, N_b=1000)

    # Set boundary data, | u, v, x, y
    boundary_data = dataloader.loading_boundary_data()
    PINN.set_boundary_data(X=boundary_data)

    # Set training data, | x, y
    training_data = dataloader.loading_training_data()
    PINN.set_eq_training_data(X=training_data)

    filename = './data/cavity_Re'+str(Re)+'_256_Uniform.mat'
    x_star, y_star, u_star, v_star = dataloader.loading_evaluate_data(filename)

    # Training
    PINN.set_alpha_evm(0.05)
    PINN.train(num_epoch=500000, lr=1e-3)
    PINN.evaluate(x_star, y_star, u_star, v_star)
#    saved_ckpt = 'model_cavity_loop%d.pth'%(0)
#    PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_f)

    PINN.set_alpha_evm(0.03)
    PINN.train(num_epoch=500000, lr=2e-4)
    PINN.evaluate(x_star, y_star, u_star, v_star)
#    saved_ckpt = 'model_cavity_loop%d.pth'%(1)
#    PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_f)

    PINN.set_alpha_evm(0.02)
    PINN.train(num_epoch=500000, lr=5e-5)
    PINN.evaluate(x_star, y_star, u_star, v_star)
#    saved_ckpt = 'model_cavity_loop%d.pth'%(2)
#    PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_f)

    PINN.set_alpha_evm(0.01)
    PINN.train(num_epoch=500000, lr=5e-5)
    PINN.evaluate(x_star, y_star, u_star, v_star)
#    saved_ckpt = 'model_cavity_loop%d.pth'%(3)
#    PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_f)

    PINN.set_alpha_evm(0.005)
    PINN.train(num_epoch=500000, lr=1e-5)
    PINN.evaluate(x_star, y_star, u_star, v_star)
#    saved_ckpt = 'model_cavity_loop%d.pth'%(4)
#    PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_f)

    PINN.set_alpha_evm(0.002)
    PINN.train(num_epoch=500000, lr=2e-6)
    PINN.evaluate(x_star, y_star, u_star, v_star)
#    saved_ckpt = 'model_cavity_loop%d.pth'%(4)
#    PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_f)

if __name__ == "__main__":
#    net_para = './simulation_04232023/results/Re4000/6x80_Nf100k_lamB100_alpha0.001/model_cavity_loop4200000.pth'
#    train(net_params= net_para)
     train()
