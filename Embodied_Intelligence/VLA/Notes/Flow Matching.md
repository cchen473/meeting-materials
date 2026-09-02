The essence of flow matching is to define a velocity field $v(x,\tau)$, such that we go from an initial distribution, along this velocity field to integrate, then we get the goal distribution.

## Mathematical Deduction: 

We start from 1-dimensional case. Assume sample points have a probability density $p_{\tau}(x)$ along the x-axis, and assume sample points move along a velocity field with $\frac{dX_{\tau}}{d\tau}=v(X_{\tau},\tau)$, so $p_{\tau}(x)$ is sample distribution at "time" $\tau$. 

Define the probability mass on interval $[a, b]$: $M_{[a,b]}(\tau) = \int_{a}^bp_{\tau}(x)dx$. Then we define the probability flux $J_{\tau}(x) = p_{\tau}(x)v(x,\tau)$, so the probability mass of points moving across $x$ around time $\tau$ is $p_{\tau}(x)v(x,\tau)d\tau$.

Then the rate of change of probability on an interval is $\frac{d}{d\tau}\int_{a}^bp_{\tau}(x)dx=J_{\tau}(a)-J_{\tau}(b)$, which means points flow in from $a$ and flow out of $b$. Since we also have $J_{\tau}(a)-J_{\tau}(b) = -\int_{a}^b\frac{\partial J_{\tau}(x)}{\partial x}dx$, combining them leads us to $\int_{a}^b[\frac{\partial p_{\tau}(x)}{\partial \tau}+\frac{\partial J_{\tau}(x)}{\partial x}]dx=0$. Since this equation works on any $[a,b]$, we prove that $\frac{\partial p_{\tau}(x)}{\partial \tau}+\frac{\partial }{\partial x}[p_{\tau}(x)v(x,\tau)]=0$. And it's easy to know $\int_{x} p_{\tau}(x)dx=1,\forall \tau$. Thus we are carrying the old probability density to form a new distribution.

Then we further expand it to $\frac{\partial p_{\tau}(x)}{\partial \tau}+\frac{\partial }{\partial x}[p_{\tau}(x)v(x,\tau)]=\frac{\partial p_{\tau}(x)}{\partial \tau}+\frac{\partial p_{\tau}(x)}{\partial x}v(x,\tau)+\frac{\partial v(x,\tau)}{\partial x}p_{\tau}(x)=0$.
Since $\frac{d}{d\tau}p_{\tau}(X_{\tau})=\frac{\partial p_{\tau}(X_{\tau})}{\partial \tau}+\frac{\partial p_{\tau}(X_{\tau})}{\partial x}\frac{dX_{\tau}}{d\tau}=\frac{\partial p_{\tau}(X_{\tau})}{\partial \tau}+\frac{\partial p_{\tau}(X_{\tau})}{\partial x}v(X_{\tau},\tau)=-\frac{\partial v(X_{\tau},\tau)}{\partial x}p_{\tau}(X_{\tau})$, we can get $\frac{d}{d\tau}\log p_{\tau}(X_{\tau})=-\frac{\partial v(X_{\tau},\tau)}{\partial x}$. So we can solve it by the initial condition and the expression of $v$. This gives us the probability density of $X_{\tau}$.

In high-dimensional case, we select any region $\Omega \subset \mathbb{R}^D$, then $M_{\Omega}(\tau)=\int_{\Omega}p_{\tau}(x)dx$, $x$ is a point in the space.

We select a small boundary $\partial \Omega$, then $\forall x \in \partial \Omega$, define unit outward normal vector $n(x)$, since velocity $v(x,\tau)$ is a vector, the velocity that go across the boundary is $v(x,\tau)\cdot n(x)$. We select a small surface $dS$, then the probability that goes across this region in $d\tau$ time is $p_{\tau}(x)[v(x,\tau)\cdot n(x)]dSd\tau$.

Then the whole pure outward probability flow is $\int_{\partial \Omega}p_{\tau}(x)[v(x,\tau)\cdot n(x)]dS$, then $\frac{d}{d\tau}\int_{\Omega}p_{\tau}(x)dx = - \int_{\partial \Omega}p_{\tau}(x)[v(x,\tau)\cdot n(x)]dS$.

We show the definition of divergence. Assume $F(x) = (F_{1}(x),\dots,F_{n}(x))$, then the divergence $\nabla \cdot F=\sum_{i=1}^n\frac{\partial F_{i}}{\partial x_{i}}$.

Divergence Theorem: $\int_{\partial \Omega}(F\cdot n)dS=\int_{\Omega}(\nabla \cdot F)dx$.

We give a proof on 2-dimensional case. Define $\Omega=[a, b]\times[c,d]$, vector field $F(x,y)=(F_{1}(x,y),F_{2}(x,y))$.

For left and right side, the unit outward normal vector at $x=b$ is $n=(1,0)$, then $F\cdot n=F_{1}(b,y)$, the flow of right boundary is $\int_{c}^dF_{1}(b,y)dy$. And the same way leads us to left boundary $-\int_{c}^dF_{1}(a,y)dy$. Sum them gives us $\int_{c}^d(F_{1}(b,y)-F_{1}(a,y))dy$. Since we also know that $F_{1}(b,y)-F_{1}(a,y)=\int_{a}^b\frac{\partial F_{1}(x,y)}{\partial x}dx$, the total flow of left and right boundary is $\int_{c}^d \int_{a}^b\frac{\partial F_{1}}{\partial x}dxdy$. Similarly, the flow of up and down boundary is $\int_{a}^b \int_{c}^d\frac{\partial F_{2}}{\partial y}dydx$. Then the whole probability flow $\int_{\partial\Omega}F\cdot ndS= \int \int\frac{\partial F_{1}}{\partial x}+\frac{\partial F_{2}}{\partial y}dxdy=\int_{\Omega}\nabla \cdot Fd\Omega$, we can also say it $\int_{\Omega}\nabla \cdot Fdx$ for consistency. Then we finish the proof.

Thus for probability flow, $F=p_{\tau}v$, $\int_{\partial \Omega}F\cdot ndS=\int_{\Omega}\nabla \cdot(p_{\tau}v)dx$, we have $\int_{\Omega}[\frac{\partial p_{\tau}}{\partial \tau}+\nabla \cdot(p_{\tau}v)]dx=0$. Since it works on every region, we have $\frac{\partial p_{\tau}}{\partial \tau}+\nabla \cdot(p_{\tau}v)=0$.

We can further expand it to get $\frac{\partial p_{\tau}}{\partial \tau}+\nabla p_{\tau}\cdot v+p_{\tau }\nabla \cdot v=0$. For the trace $X_{\tau}$, we have $\frac{d}{d\tau}p_{\tau}(X_{\tau})=\frac{\partial p_{\tau}}{\partial \tau}+\sum_{i=1}^D \frac{\partial p_{\tau}}{\partial x_{\tau,i}}\frac{dX_{\tau,i}}{d\tau}=\frac{\partial p_{\tau}}{\partial \tau}+\nabla p_{\tau} \cdot \frac{dX_{\tau}}{d\tau}=\frac{\partial p_{\tau}}{\partial \tau}+\nabla p_{\tau}\cdot v$, so this leads us to $\frac{d}{d\tau}p_{\tau}(X_{\tau})=-p_{\tau}(X_{\tau}){\nabla}\cdot v$, which is $\frac{d}{d\tau}\log p_{\tau}(X_{\tau})=-\nabla \cdot v$. Then we can easily solve this ODE if we're given $p_{0}(X_{0})$ and the velocity field.

## Application in $\pi_{0}$

In $\pi_{0}$, robots get $o_{k}=(I_{k}^1,\dots,I_{k}^n, \ell_{k},q_{k})$, where $I_{k}^i$ is $i$ -th camera image, $\ell_{k}$ is a language instruction, $q_{k}$ is the state of the robot, like joints angles.

The model wants to predict an action chunk with length $H$, $A_{k} = (a_{k}, a_{k+1},\dots,a_{k+H-1})\in \mathbb{R}^{H\times d_{a}}$, which means it wants to learn $p_{data}(A_{k}\mid o_{k})$, then we can sample from distribution $A_{k}\sim p(A_{k}\mid o_{k})$.

For simplicity we flatten $H\times d_{a}$ to $m$, so that $A_{k}\in \mathbb{R}^m$.

We sample $A\sim p_{data}(A\mid o), \epsilon\sim\mathcal{N}(0, I)$ given observation $o$, then we define $X_{\tau}=\tau A+(1-\tau)\epsilon$, so that $v=\frac{dX_{\tau}}{d\tau}=A-\epsilon$, and we have $X_{\tau}\mid A,o\sim \mathcal{N}(\tau A,(1-\tau)^2I)$. So this only gives us the conditional velocity field $v_{\tau}(X_{\tau}\mid A,o)=A-\epsilon$, but we want the marginal one.

The most direct way is to use "velocity = marginal flux / marginal density". The marginal probability flux on position $x$ is $J_{\tau}(x\mid o)=\int p_{\tau}(x\mid A,o)v_{\tau}(x\mid A,o)p_{data}(A\mid o)dA$, and the marginal probability density is $p_{\tau}(x\mid o)=\int p_{\tau}(x\mid A,o)p_{data}(A\mid o)dA$, thus $v_{\tau}(x\mid o)=\frac{J_{\tau}(x\mid o)}{p_{\tau}(x\mid o)}$.

Since $p_{\tau}(A\mid x,o)=\frac{p_{\tau}(x\mid A,o)p_{data}(A\mid o)}{p_{\tau}(x\mid o)}$, then we can reduce the expression into $v_{\tau}(x\mid o)=\int v_{\tau}(x\mid A,o)p_{\tau}(A\mid x,o)dA$, this defines $v_{\tau}(x\mid o):=\mathbb{E}[v_{\tau}(x\mid A,o)\mid X_{\tau}=x,o,\tau]$.

But we have to find a loss function to train the model. We use MSE $\mathcal{L}(\theta)=\mathbb{E}[||v_{\theta}(X_{\tau},o,\tau)-(A-\epsilon)||_{2}^2]$, which measures $v(X_{\tau},o,\tau)$, and we know that $v^*(x,o,\tau)=\mathbb{E}[A-\epsilon\mid X_{\tau}=x,o,\tau]$, so this is exactly the same with $v_{\tau}(x\mid o)$ because $v_{\tau}(x\mid A,o)=A-\epsilon$.

Then for inference stage, we sample a noise $\epsilon$, get current observation, integrate along the velocity field to output an action chunk.

I think this is the core process in $\pi_{0}$.

$E[(f(X)-\mu+\mu-Y)^2]=E[f(X)-\mu]^2+E[Y-\mu]^2+2E[(f(X)-\mu)(Y-\mu)]$
