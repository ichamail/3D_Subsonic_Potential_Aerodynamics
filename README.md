# Subsonic Potential Aerodynamics

# A 3D Aerodynamic Potential-Flow Code

## Theoretical Model

* A vector field $` \underline{V}: \mathbb{U} \to \mathbb{R}^n `$, where $` \mathbb{U} `$ is an open subset of $` \mathbb{R}^n `$, is said to be conservative if  there exists a $` \mathrm{C}^1 `$ (continuously differentiable) scalar field $` \phi `$ on $` \mathbb{U} `$ such that $` \underline{V} = \nabla \phi `$.

* According to Poincaré's Lemma, A continuously differentiable ($` \mathrm{C}^1 `$) vector field $` \underline{V} `$ defined on a simply connected subset $` \mathbb{U} `$ of $` \mathbb{R}^n `$  ($` \underline{V} \colon \mathbb{U} \subseteq \mathbb{R}^n \to \mathbb{R}^n `$), is conservative if and only if it is irrotational throughout its domain ($` \nabla \times \underline{V} = 0 `$, $` \forall \underline{x} \in \mathbb{U} `$).

* Circulation $` \Gamma = \oint_{C} \underline{V} \cdot \mathrm{d} \underline{l} = \iint_S \nabla \times \underline{V} \cdot \mathrm{d}\underline{S} `$.
* In a conservative vector field this integral evaluates to zero for every closed curve. $` \Gamma = \oint_{C} \underline{V} \cdot \mathrm{d} \underline{l} = \iint_S \nabla \times \underline{V} \cdot \mathrm{d}\underline{S} = \iint_S \nabla \times \nabla \phi \cdot \mathrm{d}\underline{S} = 0 `$ 



### Velocity field
   * incompressible: $` \nabla \cdot \underline{V} = 0 `$
   * conservative: $` \underline{V} = \nabla \phi \implies \nabla \times \underline{V} = \nabla \times \nabla \phi = 0 `$ (irrotational)

```math
\nabla \cdot \underline{V} = \nabla \cdot \nabla \phi = \nabla^2 \phi = 0 \qquad \text{(Laplace's Equation)}
```

### Rotational Invariance of Laplace differential operator
A function defined on an inner product space is said to have rotational invariance if its value does not change when arbitrary rotations are applied to its argument. 
$` f(\underline{x}') = f(\mathbf{R}\underline{x}) = f(\underline{x}) `$ This also applies for an operator that acts on a function $` f : \mathbb{U} \subseteq \mathbb{R} \to \mathbb{U} `$.
In that case rotational invariance may also mean that the function commutes with rotations of elements in $` \mathbb{U} `$. An example is the Laplace differential operator 
$` \nabla^2 = \frac{\partial^2()}{\partial x^2} + \frac{\partial^2()}{\partial y^2} + \frac{\partial^2()}{\partial z^2} `$.


$` \nabla_\mathbf{x}^2 u(\mathbf{x}) = \nabla_\mathbf{y}^2 v(\mathbf{y}) `$, where $` \mathbf{y} = \mathbf{R} \mathbf{x} `$ 
and $`  u(\mathbf{x}) = v(\mathbf{y}) = v(\mathbf{R} \mathbf{x}) `$

### Fundamental Solution of the Laplace operator
Since  Laplace equation is invariant under rigid motions, it is natural to look for solutions to $` \nabla^2 \psi = 0 `$ which have rotational symmetry and the form 
$` \psi \colon \mathbb{R}^3 \to \mathbb{R} \colon \psi(\underline{r},\underline{r}_p) = \psi(\lVert \underline{r} - \underline{r}_p \rVert) `$

Assuming that $` \underline{r} \neq \underline{r}_p `$ , it is true that

```math
\begin{align*}
&\frac{\partial \psi}{\partial x} = \psi'(\lVert \underline{r} - \underline{r}_p \rVert) \frac{x-x_p}{\lVert \underline{r} - \underline{r}_p \rVert} \, , \qquad
\frac{\partial \psi}{\partial x} = \psi'(\lVert \underline{r} - \underline{r}_p \rVert) \frac{x-x_p}{\lVert \underline{r} - \underline{r}_p \rVert} \, , \qquad
\frac{\partial \psi}{\partial x} = \psi'(\lVert \underline{r} - \underline{r}_p \rVert) \frac{x-x_p}{\lVert \underline{r} - \underline{r}_p \rVert}
\\
&\frac{\partial^2 \psi}{\partial x^2} = \frac{(x-x_p)^2}{\lVert \underline{r} - \underline{r}_p \rVert^2} \psi''(\lVert \underline{r} - \underline{r}_p \rVert)
+
\frac{1}{\lVert \underline{r} - \underline{r}_p \rVert} \psi'(\lVert \underline{r} - \underline{r}_p \rVert)
-
\frac{(x-x_p)^2}{\lVert \underline{r} - \underline{r}_p \rVert^3} \psi'(\lVert \underline{r} - \underline{r}_p \rVert)
\\
&\frac{\partial^2 \psi}{\partial y^2} = \frac{(y-y_p)^2}{\lVert \underline{r} - \underline{r}_p \rVert^2} \psi''(\lVert \underline{r} - \underline{r}_p \rVert)
+
\frac{1}{\lVert \underline{r} - \underline{r}_p \rVert} \psi'(\lVert \underline{r} - \underline{r}_p \rVert)
-
\frac{(y-y_p)^2}{\lVert \underline{r} - \underline{r}_p \rVert^3} \psi'(\lVert \underline{r} - \underline{r}_p \rVert)
\\
&\frac{\partial^2 \psi}{\partial z^2} = \frac{(z-z_p)^2}{\lVert \underline{r} - \underline{r}_p \rVert^2} \psi''(\lVert \underline{r} - \underline{r}_p \rVert)
+
\frac{1}{\lVert \underline{r} - \underline{r}_p \rVert} \psi'(\lVert \underline{r} - \underline{r}_p \rVert)
-
\frac{(z-z_p)^2}{\lVert \underline{r} - \underline{r}_p \rVert^3} \psi'(\lVert \underline{r} - \underline{r}_p \rVert)
\end{align*}
```

```math
\nabla^2 \psi = \psi''(\lVert \underline{r} - \underline{r}_p \rVert) - \frac{2}{\lVert \underline{r} - \underline{r}_p \rVert} \psi'(\lVert \underline{r} - \underline{r}_p \rVert) = 0
\qquad \forall \underline{r} \in \mathbb{R}^3 - \{\underline{r}_p\}
```

```math
\begin{align*}
&\psi''(\lVert \underline{r} - \underline{r}_p \rVert) - \frac{2}{\lVert \underline{r} - \underline{r}_p \rVert} \psi'(\lVert \underline{r} - \underline{r}_p \rVert) = 0 \implies
\\
&\lVert \underline{r} - \underline{r}_p \rVert^2 \psi''(\lVert \underline{r} - \underline{r}_p \rVert)
- 2 \lVert \underline{r} - \underline{r}_p \rVert \psi'(\lVert \underline{r} - \underline{r}_p \rVert)
= 0
\implies
\left[ \lVert \underline{r} - \underline{r}_p \rVert^2 \psi'(\lVert \underline{r} - \underline{r}_p \rVert) \right]' = 0
\implies
\\
&\psi'(\lVert \underline{r} - \underline{r}_p \rVert) =
\frac{c}{\lVert \underline{r} - \underline{r}_p \rVert^2}
\implies
\psi(\lVert \underline{r} - \underline{r}_p \rVert) = - \frac{c}{\lVert \underline{r} - \underline{r}_p \rVert} + c' \, , \qquad c, c' \in \mathbb{R}
\end{align*}
```


When  $` c = \frac{1}{4 \pi} `$ and $` c' = 0 `$ are chosen, so that
$` \psi(\lVert \underline{r} - \underline{r}_p \rVert) = - \frac{1}{4 \pi} \frac{1}{\lVert \underline{r} - \underline{r}_p \rVert} `$, it can be shown that 

```math
\nabla^2 \psi(\underline{r}, \underline{r}_p) = \delta(\underline{r} - \underline{r}_p)
\qquad \text{and} \qquad \iiint_V f(\underline{r}) \delta(\underline{r} - \underline{r}_p) \mathrm{d}V = f(\underline{r}_p)
```

where $` \delta(\underline{r} - \underline{r}_p) `$ is the Dirac delta function defined in $` \mathrm{C}_c^\infty(V) `$ and, $` f \colon V \subseteq \mathbb{R}^3 \to \mathbb{R} `$ and $` f \in \mathrm{C}_c^\infty(V) `$.


$` \psi(\lVert \underline{r} - \underline{r}_p \rVert) = - \frac{1}{4 \pi} \frac{1}{\lVert \underline{r} - \underline{r}_p \rVert} `$ is called Fundamental Solution of the Laplace operator


### Gauss's Divergence Theorem

Let $` V \subset \mathbb{R}^3 `$ be a bounded domain, and his boundary $` \partial V `$.
Let $` \partial V `$ be a smooth hypersurface and $` \underline{n} `$ the outward unit normal vector to $` \partial V `$.
Supose $` \underline{F} \colon V \subseteq \mathbb{R}^3 \to \mathbb{R}^3 `$ and  $` F \in \mathrm{C}^1(V) \cap \mathrm{C}^0(\partial V) `$. It is true that

```math
\iiint_V \nabla \cdot \vec{F} \mathrm{d}V = \iint_{\partial V} \vec{F} \cdot \vec{n} \mathrm{d}S
```

### Green's 2nd Identity
Let $` V \subset \mathbb{R}^3 `$ be a bounded domain, and his boundary $` \partial V `$.
Let $` \partial V `$ be a smooth hypersurface and $` \underline{n} `$ the outward unit normal vector to $` \partial V `$.
Supose $` \underline{F} = \psi \nabla \phi - \phi \nabla \psi `$, 
where $` \psi \, , \phi \in \mathrm{C}^2(V) \cap \mathrm{C}^1(\partial V) `$. It is true that

```math
\begin{align*}
&\iiint_V \nabla \cdot \vec{F} \mathrm{d}V = \iint_{S} \vec{F} \cdot \vec{n} \mathrm{d}S
\implies
\iiint_V \nabla \cdot \left( \psi \nabla \phi - \phi \nabla \psi \right) \mathrm{d}V = 
\iint_{S} \left( \psi \nabla \phi - \phi \nabla \psi \right) \cdot \vec{n} \mathrm{d}S
\implies
\\
&\iiint_V \left( \nabla \psi \cdot \nabla \phi + \psi \nabla^2 \phi  - \nabla \phi \cdot \nabla \psi - \phi \nabla^2 \psi \right) \mathrm{d}V = 
\iint_{S} \left[ \psi (\vec{n} \cdot \nabla) \phi - \phi (\vec{n} \cdot \nabla) \psi \right] \mathrm{d}S
\implies
\end{align*}
```
```math
\begin{aligned}
\iiint_V \left(\psi \nabla^2 \phi - \phi \nabla^2 \psi \right) \mathrm{d}V &=
\iint_{S} \left[ \psi (\vec{n} \cdot \nabla) \phi - \phi (\vec{n} \cdot \nabla) \psi \right] \mathrm{d}S
\\
&= \iint_{S} \left( \psi  \frac{\partial \phi}{\partial n} - \phi \frac{\partial \psi}{\partial n} \right)\mathrm{d}S
\end{aligned}
```



### Integral Equation of velocity potential $\phi$
* Let $` V \subset \mathbb{R}^3 `$ be a bounded domain, and his boundary $` \partial V `$.
* Let $` \partial V =  S_\infty \cup S \cup S_w `$ be a smooth hypersurface and $` \underline{e}_n (= - \underline{n} ) `$ the inward unit normal vector to $` \partial V `$.
* Let $` \phi \in \mathrm{C}^2(V) \cap \mathrm{C}^1(\partial V) `$ and $` \psi(\lVert \underline{r} - \underline{r}_p \rVert) = - \frac{1}{4 \pi} \frac{1}{\lVert \underline{r} - \underline{r}_p \rVert} `$
* Let $` V_\epsilon = V - B[\underline{r}_p, \epsilon] `$. Then $` \partial V_\epsilon = \partial V \cup \partial B[\vec{r}_p, \epsilon] = S_\infty \cup S \cup S_w \cup S_\epsilon `$
* Let Velocity field $`\underline{V} = \nabla \phi `$  and $` \nabla \cdot \underline{V} = 0 `$. Then $` \nabla^2 \phi = 0 `$


Using Green's 2nd Identity we have
```math
\begin{align*}
&\iiint_{V_\epsilon} \left(\psi \nabla^2 \phi - \phi \nabla^2 \psi \right) \mathrm{d}V =
\iint_{\partial V_\epsilon} \left[ \psi (\underline{n} \cdot \nabla) \phi - \phi (\underline{n} \cdot \nabla) \psi \right] \mathrm{d}S \xRightarrow[\nabla^2 \phi = 0]{\nabla^2 \psi(\underline{r}, \underline{r}_p) = 0 \, , \underline{r} \neq \underline{r}_p}
\\
&
\iint_{\partial V_\epsilon} \left[ \psi (\underline{n} \cdot \nabla) \phi - \phi (\underline{n} \cdot \nabla) \psi \right] \mathrm{d}S = 0
\xRightarrow{\underline{e}_n = - \underline{n}}
\iint_{\partial V_\epsilon} \left[ \phi (\vec{e}_n \cdot \nabla) \psi - \psi (\vec{e}_n \cdot \nabla) \phi \right] \mathrm{d}S = 0
\xRightarrow{\partial V_\epsilon = \partial V \cup \partial B[\vec{r}_p, \epsilon]}
\end{align*}
```

```math
\iint_{\partial V} \left[ \phi (\vec{e}_n \cdot \nabla) \psi - \psi (\vec{e}_n \cdot \nabla) \phi \right] \mathrm{d}S
+
\iint_{\partial B[\vec{r}_p, \epsilon]} \left[ \phi (\vec{e}_n \cdot \nabla) \psi - \psi (\vec{e}_n \cdot \nabla) \phi \right] \mathrm{d}S
= 0
```

```math
\phi(\underline{r}_p) =
\iint_S \frac{\sigma}{2\pi} \ln{(\lVert \underline{r} - \underline{r}_p \rVert)} \mathrm{d}S + \iint_S - \frac{\mu}{2\pi} (\underline{e}_n \cdot \nabla) \ln{(\lVert \underline{r} - \underline{r}_p \rVert)} \mathrm{d}S + \iint_{S_w} - \frac{\mu_w}{2\pi} (\underline{e}_n \cdot \nabla) \ln{(\lVert \underline{r} - \underline{r}_p \rVert)} \mathrm{d}S + \phi_\infty(\underline{r}_p)
```



where:
   * $` \mu = \phi - \phi_i, \mu_w = \phi_U - \phi_L `$
   * $` \sigma = (\underline{e}_n \cdot \nabla)(\phi - \phi_i) `$
   * $` \phi_\infty(\underline{r}_p) = \iint_{S_\infty}  \left[ (\underline{e}_n \cdot \nabla) \phi \ln{ \left( \frac{\lVert \underline{r} - \underline{r}_p \rVert}{2\pi} \right) } - \phi  (\underline{e}_n \cdot \nabla) \ln{ \left( \frac{\lVert \underline{r} - \underline{r}_p \rVert}{2\pi} \right) } \right] \mathrm{d}S `$


if $` \mu = \phi - \phi_i = \phi - \phi_\infty `$ and $` \sigma = (\underline{e}_n \cdot \nabla)(\phi - \phi_i) = (\underline{e}_n \cdot \nabla)(\phi - \phi_\infty ) `$, then for $` P \in S^- `$ :

```math
\iint_S \frac{\sigma}{2\pi} \ln{(\lVert \underline{r} - \underline{r}_p \rVert)} \mathrm{d}S + \iint_S - \frac{\mu}{2\pi} (\underline{e}_n \cdot \nabla) \ln{(\lVert \underline{r} - \underline{r}_p \rVert)} \mathrm{d}S + \iint_{S_w} - \frac{\mu_w}{2\pi} (\underline{e}_n \cdot \nabla) \ln{(\lVert \underline{r} - \underline{r}_p \rVert)} \mathrm{d}S = 0, \hspace{0.5cm} \forall (x_p, y_p, z_p) \in S: (\underline{r} - \underline{r}_p) \cdot \underline{e}_n \to 0^+
```

note: $` \sigma = (\underline{e}_n \cdot \nabla)(\phi - \phi_i) = (\underline{e}_n \cdot \nabla)(\phi - \phi_\infty ) = (\underline{e}_n \cdot \nabla)\phi - (\underline{e}_n \cdot \nabla)\phi_\infty = \underline{V} - \underline{e}_n \cdot \underline{V}_\infty = - \underline{e}_n \cdot \underline{V}_\infty `$

Notable Observations:
   * A vector field $` \underline{V}: U \to \mathbb{R}^n `$, where $` \mathbb{U} `$ is an open subset of $` \mathbb{R}^n `$, is said to be conservative if  there exists a $` \mathrm{C}^1 `$ (continuously differentiable) scalar field $` \phi `$ on $` \mathbb{U} `$ such that $` \underline{V} = \nabla \phi `$.

   * According to Poincaré's Lemma, A continuously differentiable ($` \mathrm{C}^1 `$) vector field $` \underline{V} `$ defined on a simply connected subset $` \mathbb{U} `$ of $` \mathbb{R}^n `$  ($` \underline{V} \colon \mathbb{U} \subseteq \mathbb{R}^n \to \mathbb{R}^n `$), is conservative if and only if it is irrotational throughout its domain ($` \nabla \times \underline{V} = 0 `$, $` \forall \underline{x} \in \mathbb{U} `$).

   * Circulation $` \Gamma = \oint_{C} \underline{V} \cdot \mathrm{d} \underline{l} = \iint_S \nabla \times \underline{V} \cdot \mathrm{d}\underline{S} `$ 
   In a conservative vector field this integral evaluates to zero for every closed curve. $` \Gamma = \oint_{C} \underline{V} \cdot \mathrm{d} \underline{l} = \iint_S \nabla \times \underline{V} \cdot \mathrm{d}\underline{S} = \iint_S \nabla \times \nabla \phi \cdot \mathrm{d}\underline{S} = 0 `$

   * The space exterior to a 2D object (e.g. an airfoil) is not simply connected

### Numerical Model (Panel Methods)

```math
B_{ij} \sigma_j + C_{ij} \mu_j = 0 , \qquad 0 \le i < N_s \qquad and  \qquad 0 \le j < N_s + N_w 
``` 

where:
   * $` B_{ij} =  \frac{1}{2\pi} \iint_{S_j}  \ln{(\lVert \underline{r} - \underline{r}_{cp_i} \rVert)} \mathrm{d}{S_j}  = \frac{1}{2\pi} \iint_{t_1}^{t_2}  \ln \left(\sqrt{(t_{cp_i} - t)^2 + n_{cp_i}^2} \right) \mathrm{d}{t_j} `$

   * $` C_{ij} =  \frac{1}{2\pi} \iint_{S_j}  (\underline{e}_n \cdot \nabla) \ln{(\lVert \underline{r} - \underline{r}_{cp_i} \rVert)} \mathrm{d}{S_j} = \frac{1}{2\pi} \iint_{S_j}  \frac{n_{cp_i}}{(t_{cp_i} - t)^2 + n_{cp_i}^2} \mathrm{d}{t_j} `$

from Kutta Condition: $` \mu_w = const = \mu_U - \mu_L `$
```math
A_{ij} \mu_j = - B_{ij} \sigma_j , \hspace{0.5cm} A_{ij} = 
\begin{cases}
   C_{ij} + \sum\limits_{k} C_{ik} & \text{if $j=0$}\\
   C_{ij} & \text{if $0 < j < N_s - 1$}\\
   C_{ij} - \sum\limits_{k} C_{ik} & \text{if $j=N_s-1$}
\end{cases} 
```


```math
0 \le i < N_s  \qquad 0 \le j < N_s  \qquad N_s \le k < N_s + N_w
```

### Features
