# NSFnet
                          \textbf{PINNs find a new type of solution for the 2D steady cavity flow} 

A plurality of stable and unstable solutions of the 2D cavity flow at $\boldsymbol{Re=2,000}$. Shown are inference streamlines by the original NSFnet and the entropy-viscosity regularized ev-NSFnet. The five flow plots on each side of the DNS (Reference) solution are obtained based on five independent initialization values of PINNs using the same hyper-parameters. A1 and A2 represent a new flow type (Class 1) that is not captured by DNS or by ev-NSFnet. Adding eddy viscosity to the Navier-Stokes leads to a stable solution, similar with what is captured by DNS. The average error in the velocity field for B1-B5 is less than 4\%.
![](https://github.com/Scien42/NSFnet/blob/main/resources/ev_NSFnet.png)

Class 1 | Class 2
--- | ---
![](https://github.com/Scien42/NSFnet/blob/main/Re2K_class1.gif) | ![](https://github.com/Scien42/NSFnet/blob/main/resources/Re2K_Class2.gif) 








