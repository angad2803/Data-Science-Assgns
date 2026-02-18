# Linear Programming in MATLAB

### Lab Evaluation Study Notes | Feb 16–20

---

## 📌 What You Need to Know

| Topic                        | Marks | Task                           |
| ---------------------------- | ----- | ------------------------------ |
| Graphical / Algebraic Method | 5     | Solve manually on paper        |
| MATLAB Coding                | 7     | Write & run code, get solution |

---

---

# PROGRAM 1 — Basic Feasible Solution (Algebraic / BFS Method)

## The Problem

**Maximize:**

```
Z = 2x₁ + 3x₂ + 4x₃ + 7x₄
```

**Subject to:**

```
2x₁ + 3x₂ − x₃ + 4x₄ = 8
 x₁ − 2x₂ + 6x₃ − 7x₄ = −3
x₁, x₂, x₃, x₄ ≥ 0
```

---

## The Code (Phase by Phase)

### 🔷 Phase 1 — Input the Parameters

```matlab
clc
clear all
format short

A = [2, 3, -1, 4;   % Coefficient matrix (constraints)
     1, -2, 6, -7]

C = [2, 3, 4, 7]    % Cost coefficients (objective function)

b = [8; -3]          % Right-hand side values
```

> **What is A?** Each row = one constraint. Each column = one variable (x₁, x₂, x₃, x₄).

---

### 🔷 Phase 2 — Count Variables and Constraints

```matlab
n = size(A, 2)   % Number of variables  → n = 4
m = size(A, 1)   % Number of constraints → m = 2
```

> `size(A, 2)` = number of **columns** (variables)
> `size(A, 1)` = number of **rows** (constraints)

---

### 🔷 Phase 3 — Generate All Basic Solutions (nCm Combinations)

```matlab
if (n > m)
    nCm  = nchoosek(n, m)       % Total combinations = C(4,2) = 6
    pair = nchoosek(1:n, m)     % Each row is one pair of column indices
```

> We pick **m columns** out of **n** to form a square basis matrix.
> Each pair gives one basic solution.

---

### 🔷 Phase 4 & 5 — Build & Check Feasibility

```matlab
sol = [];              % Start with empty solution set

for i = 1:nCm
    y = zeros(n, 1)                  % Initialize all variables to 0
    x = A(:, pair(i,:)) \ b         % Solve the basis: Bx = b

    % Keep only if all values ≥ 0 (feasibility check)
    if all(x >= 0 & x ~= inf & x ~= -inf)
        y(pair(i,:)) = x             % Put solution back in full vector
        sol = [sol, y]               % Collect feasible solutions
    end
end
```

> **`\` (backslash)** solves the linear system — it's MATLAB's way of doing `B⁻¹ × b`.
> We only keep solutions where **all variables ≥ 0** (non-negative = feasible).

---

### 🔷 Phase 6 — Evaluate Objective Function

```matlab
Z = C * sol              % Compute Z for each basic feasible solution

[Zmax, Zindex] = max(Z)  % Find maximum Z and which solution gives it
bfs = sol(:, Zindex)     % Extract the optimal BFS
```

---

### 🔷 Phase 7 — Display the Result

```matlab
optimal_value = [bfs' Zmax]
optimal_bfs   = array2table(optimal_value)
optimal_bfs.Properties.VariableNames(1:size(optimal_bfs,2)) = ...
    {'x_1','x_2','x_3','x_4','Z'}
```

> This prints a neat table showing the optimal values of x₁, x₂, x₃, x₄ and Z.

---

## 🗺️ Program 1 — Flowchart (Write This on Paper!)

```
START
  │
  ▼
Input: A, C, b
  │
  ▼
n = cols of A,  m = rows of A
  │
  ▼
Generate all C(n,m) column pairs
  │
  ▼
For each pair:
  ├─ Solve Bx = b
  ├─ All x ≥ 0?  ──No──▶ Skip
  └─ Yes ──▶ Save as BFS
  │
  ▼
Z = C × [all BFS]
  │
  ▼
Find max(Z)  →  Optimal BFS
  │
  ▼
Print table
  │
STOP
```

---

---

# PROGRAM 2 — Graphical Method (2-Variable LP)

## The Problem

**Maximize:**

```
Z = 3x₁ + 5x₂
```

**Subject to:**

```
x₁ + 2x₂ ≤ 2000
x₁ +  x₂ ≤ 1500
       x₂ ≤  600
x₁, x₂ ≥ 0
```

---

## The Code (Phase by Phase)

### 🔷 Phase 1 — Input the Parameters

```matlab
format short
clear all
clc

C = [3 5];                    % Objective function coefficients
A = [1 2; 1 1; 0 1];         % Constraint matrix (3 constraints, 2 variables)
b = [2000; 1500; 600];        % Right-hand side
```

---

### 🔷 Phase 2 — Plot the Constraint Lines

```matlab
y1 = 0:1:max(b);              % x₁ values from 0 to max(b)

% Rearranging each constraint for x₂:  x₂ = (b - a₁x₁) / a₂
x21 = (b(1) - A(1,1).*y1) ./ A(1,2);   % Constraint 1
x22 = (b(2) - A(2,1).*y1) ./ A(2,2);   % Constraint 2
x23 = (b(3) - A(3,1).*y1) ./ A(3,2);   % Constraint 3

% Clip to first quadrant (no negative values)
x21 = max(0, x21);
x22 = max(0, x22);
x23 = max(0, x23);

% Draw the graph
plot(y1, x21, 'r', y1, x22, 'k', y1, x23, 'b')
xlabel('value of x1')
ylabel('value of x2')
legend('x1 + 2x2 = 2000', 'x1 + x2 = 1500', 'x2 = 600')
grid on
```

> Each constraint line is drawn by rearranging it into `x₂ = (b − a₁x₁)/a₂` form.
> `max(0, ...)` clips below-zero values so lines stay in the first quadrant.

---

### 🔷 Phase 3 — Find Axis Intercepts (Corner Points on Axes)

```matlab
cx1 = find(y1 == 0);          % Index where x₁ = 0 (y-axis)
c1  = find(x21 == 0);         % Index where constraint 1 hits x-axis
Line1 = [y1(:,[c1 cx1]); x21(:,[c1 cx1])]';

c2 = find(x22 == 0);
Line2 = [y1(:,[c2 cx1]); x22(:,[c2 cx1])]';

c3 = find(x23 == 0);
Line3 = [y1(:,[c3 cx1]); x23(:,[c3 cx1])]';

corpt = unique([Line1; Line2; Line3], 'rows');
```

> We collect where each line crosses the x-axis and y-axis — these are corner points.

---

### 🔷 Phase 4 — Find Intersections Between Constraint Lines

```matlab
pt = [0; 0];

for i = 1:size(A,1)
    for j = i+1:size(A,1)
        A1 = [A(i,:); A(j,:)];     % Stack two constraint rows
        B1 = [b(i); b(j)];
        X  = A1 \ B1;              % Solve for intersection point
        pt = [pt, X];              % Collect all intersection points
    end
end

ptt = pt';
```

> For every **pair** of constraints, we solve the 2×2 system to get their crossing point.

---

### 🔷 Phase 5 — Collect All Corner Points

```matlab
allpt  = [ptt; corpt];
points = unique(allpt, 'rows');   % Remove duplicates
```

---

### 🔷 Phase 6 — Filter: Keep Only Feasible Points

```matlab
for i = 1:size(points,1)
    const1(i) = A(1,1)*points(i,1) + A(1,2)*points(i,2) - b(1);
    const2(i) = A(2,1)*points(i,1) + A(2,2)*points(i,2) - b(2);
    const3(i) = A(3,1)*points(i,1) + A(3,2)*points(i,2) - b(3);
end

% Find points that VIOLATE any constraint (value > 0 means outside boundary)
s1 = find(const1 > 0);
s2 = find(const2 > 0);
s3 = find(const3 > 0);

S = unique([s1 s2 s3]);
points(S,:) = [];     % Remove infeasible points
```

> A point is **infeasible** if it makes any constraint value positive (i.e., violates `≤`).

---

### 🔷 Phase 7 — Compute Z and Find Maximum

```matlab
value = points * C';              % Z = 3x₁ + 5x₂ for each feasible point
table = [points value]

[obj, index] = max(value);        % Best Z value and its index
X1 = points(index, 1);
X2 = points(index, 2);

fprintf('Objective value is %f at (%f, %f)', obj, X1, X2);
```

---

## 🗺️ Program 2 — Flowchart (Write This on Paper!)

```
START
  │
  ▼
Input: C, A, b
  │
  ▼
Rearrange constraints → x₂ = f(x₁)
Plot all constraint lines
  │
  ▼
Find axis intercepts for each line
  │
  ▼
Find intersections between every pair of lines
  │
  ▼
Combine all corner points → remove duplicates
  │
  ▼
For each point: check all constraints
Remove infeasible points (those outside the region)
  │
  ▼
Compute Z = C × xᵀ for each feasible point
  │
  ▼
Find max(Z)  →  Print optimal (x₁, x₂) and Z
  │
STOP
```

---

---

# 📝 Key Concepts Cheat Sheet

## Terminology

| Term                | Meaning                                                            |
| ------------------- | ------------------------------------------------------------------ |
| **Basic Solution**  | A solution obtained by setting (n−m) variables to zero and solving |
| **BFS**             | Basic Feasible Solution — a basic solution where all variables ≥ 0 |
| **n**               | Number of decision variables                                       |
| **m**               | Number of constraints                                              |
| **nCm = C(n,m)**    | Number of ways to choose m items from n                            |
| **Basis**           | The m columns chosen to form a square, solvable matrix             |
| **Feasible Region** | Set of all points satisfying every constraint                      |
| **Corner Point**    | A vertex of the feasible region (optimal is always here!)          |

## Important MATLAB Commands

| Command               | What It Does                           |
| --------------------- | -------------------------------------- |
| `size(A, 1)`          | Number of rows                         |
| `size(A, 2)`          | Number of columns                      |
| `nchoosek(n, m)`      | Value of C(n, m)                       |
| `nchoosek(1:n, m)`    | All combinations as rows               |
| `A \ b`               | Solve linear system Ax = b             |
| `all(x >= 0)`         | Check if all elements are non-negative |
| `zeros(n, 1)`         | Column vector of zeros                 |
| `max(Z)`              | Returns [max\_value, index]            |
| `find(x == 0)`        | Indices where condition is true        |
| `unique(..., 'rows')` | Remove duplicate rows                  |
| `array2table(...)`    | Convert matrix to readable table       |
| `fprintf(...)`        | Print formatted output                 |

## Quick Formulas

```
Number of Basic Solutions  = C(n, m) = n! / (m! × (n−m)!)

Constraint line (for graph): x₂ = (b − a₁x₁) / a₂

Objective function:  Z = C × x
```

---

_Good luck on your lab evaluation! 🎯_
