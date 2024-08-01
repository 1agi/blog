<br><br>
> Some mathematical concepts that will come in handy while understanding Machine Learning.

<br><br>
# Linear Algebra

### 1. Vectors
Imagine a vector as a list of instructions for how to move in different directions. If you're in a video game, a vector could tell you how many steps to move forward, how many to move right, and how many to move up. Adding vectors is like combining these instructions, and multiplying by a number is like repeating the instructions that many times.


<br>
<br>
A vector is an ordered list of numbers
- Represented as column vectors: $\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$
<br>Vector operations:
  <br> Addition: $\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_n + b_n \end{bmatrix}$ <br>
  <br> Scalar multiplication: $c\mathbf{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{bmatrix}$ <br>
  <br> Dot product: $\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i$

<br><br>

## 2. Matrices
Think of a matrix as a table of numbers. Each number has a specific place, like in a spreadsheet. When we do things with matrices, we're working with all these numbers at once. Adding matrices is like adding the numbers in the same spots in two different tables. Multiplying matrices is a bit trickier, but it's like taking one table's rows and another table's columns and mixing them up in a special way.

<br>

<br>
<br> A matrix is a rectangular array of numbers
- Represented as: $A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$
<br> Matrix operations:
  <br> Addition: $(A + B)_{ij} = A_{ij} + B_{ij}$
  <br> Scalar multiplication: $(cA)_{ij} = cA_{ij}$
  <br> Matrix multiplication: $(AB)_{ij} = \sum_{k=1}^n A_{ik}B_{kj}$

<br><br>

## 3. Eigenvalues and Eigenvectors
Imagine you have a special machine (the matrix) that changes things. An eigenvector is like a magic object that, when put into this machine, comes out looking exactly the same but maybe bigger or smaller. The eigenvalue tells you how much bigger or smaller it got. These are super important because they help us understand how the machine (matrix) works in a simpler way.


- For a square matrix $A$, if $A\mathbf{v} = \lambda\mathbf{v}$, then:
  - $\lambda$ is an eigenvalue of $A$
  - $\mathbf{v}$ is an eigenvector of $A$
- Characteristic equation: $det(A - \lambda I) = 0$
- Eigenvectors are linearly independent

<br><br>

## 4. Matrix Decomposition
Matrix decomposition is like taking apart a complicated toy to see how it works. SVD is a way to break any matrix into three simpler parts, which helps us understand what the matrix does. Eigendecomposition is similar but only works for special matrices - it's like finding the secret code that makes the matrix tick.
- Singular Value Decomposition (SVD): $A = U\Sigma V^T$
  - $U$ and $V$ are orthogonal matrices
  - $\Sigma$ is a diagonal matrix of singular values
- Eigendecomposition: $A = Q\Lambda Q^{-1}$
  - $Q$ is a matrix of eigenvectors
  - $\Lambda$ is a diagonal matrix of eigenvalues

<br><br>

## 5. Linear Transformations
A linear transformation is like a machine that changes vectors in a consistent way. If you put in two vectors and add the results, it's the same as adding the vectors first and then putting them in the machine. It's also like a fair machine - if you make your input twice as big, the output will be exactly twice as big too.

- A function $T: \mathbb{R}^n \to \mathbb{R}^m$ is a linear transformation if:
  1. $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
  2. $T(c\mathbf{v}) = cT(\mathbf{v})$
- Every linear transformation can be represented by a matrix

<br><br>

## 6. Vector Spaces and Subspaces
A vector space is like a playground for vectors where they can be added together or made bigger or smaller. A subspace is a special part of this playground where if you stay inside and play with the vectors, you'll always end up with another vector in the same area. The span is all the places you can reach by combining certain vectors in different ways.
<br>
- A vector space is a set $V$ with operations $+$ and $\cdot$ that satisfy certain axioms
- A subspace is a subset $H$ of a vector space $V$ that is closed under addition and scalar multiplication
- Span of vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$: $span(\mathbf{v}_1, \ldots, \mathbf{v}_k) = \{c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k : c_1, \ldots, c_k \in \mathbb{R}\}$

<br><br>


## 7. Basis and Dimension
A basis is like a set of building blocks that you can use to make any vector in your space. The dimension tells you how many different types of blocks you need. For example, in a 3D world, you need three directions (up/down, left/right, forward/backward) to describe any position, so the dimension is 3.

<br>
A basis is a linearly independent set of vectors that spans the vector space
<br> The dimension of a vector space is the number of vectors in a basis
<br> Standard basis for $\mathbb{R}^n$: $\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix}, \mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \\ \vdots \\ 0 \end{bmatrix}, \ldots, \mathbf{e}_n = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 1 \end{bmatrix}$
<br><br>

## 8. Orthogonality and Projections
Orthogonal vectors are like perpendicular lines - they're at right angles to each other. Projection is like shining a light from one vector onto another and seeing its shadow. It helps us break down vectors into parts that are easier to work with.
- Vectors $\mathbf{u}$ and $\mathbf{v}$ are orthogonal if $\mathbf{u} \cdot \mathbf{v} = 0$
- Projection of $\mathbf{b}$ onto $\mathbf{a}$: $proj_\mathbf{a}\mathbf{b} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|^2}\mathbf{a}$
- Gram-Schmidt process for orthogonalization

<br>

## 9. Determinants
The determinant is like a special number that tells you important information about a matrix. It's a bit like finding the area of a shape, but for matrices. If the determinant is zero, it means the matrix squishes everything into a lower dimension, like flattening a 3D object into 2D.

- For a 2x2 matrix: $det\begin{pmatrix}a & b \\ c & d\end{pmatrix} = ad - bc$
- Properties:
  1. $det(AB) = det(A)det(B)$
  2. $det(A^T) = det(A)$
  3. $det(A^{-1}) = \frac{1}{det(A)}$ if $A$ is invertible

<br>


## 10. Linear Systems and Gaussian Elimination
Solving a system of linear equations is like solving a puzzle where each piece affects the others. Gaussian elimination is a step-by-step way to solve this puzzle. It's like untangling a bunch of strings one at a time until you can see clearly what each string (variable) should be.

These concepts form the foundation of linear algebra and are crucial for understanding many machine learning algorithms and techniques.

- System of linear equations: $A\mathbf{x} = \mathbf{b}$
- Gaussian elimination steps:
  1. Convert to augmented matrix $[A|\mathbf{b}]$
  2. Use elementary row operations to get row echelon form
  3. Back-substitute to solve for variables
- Solutions: unique, infinitely many, or none

<br><br>

