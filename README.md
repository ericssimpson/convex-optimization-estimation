# Linear Program Feasibility Project

**Overview**

This project investigates innovative applications of deep learning for linear programming (LP) optimization. The core focus is on developing a model to rapidly estimate the feasibility of a linear program, potentially accelerating traditional LP solving algorithms.

[Image of sample linear program visualization] 

**Problem**

Determining whether a linear program has a feasible region is a crucial step in LP solving. Traditional methods like the simplex algorithm can be computationally expensive, especially for larger problems.

**Results**

Initial results on 2D linear programs are highly promising. The model demonstrates:

*   High training accuracy (above 95%)
*   Robust performance during cross-validation, even on datasets with varying feasibility biases.

[Image of training loss/accuracy graph] 

**Current Functionality**

* **Linear Program Generation:** The code can generate random LPs with customizable dimensions and coefficient bounds.
* **Machine Learning Feasibility Prediction:**  A simple neural network model is trained to predict LP feasibility based on the structure of the LP (A, b, and c matrices).
* **Visualization:** Contains capabilities for plotting linear inequalities, convex feasible regions, and interior points.

[Image of a visualized feasible region]

**Data Generation**

The `create_balanced_data` function generates balanced training data. Ideally, this function should work to eliminate biasing attributes in the generated linear programs to prevent overfitting to artificial patterns. Currently, it uses simplified randomization techniques. The intent is to explore more sophisticated bias reduction methods in the future.

**Machine Learning Model**

The current model is a simple neural network consisting of Dense layers. Input data, representing the LP's structure, is preprocessed and normalized before being fed to the model.

**Work in Progress**

* **Dynamic LP Handling:** Development is underway to enable the handling of  linear programs defined with an arbitrary number of variables and constraints.

**How to Use (Basic)**

1. **Install Dependencies:**
   ```bash
   pip install numpy scipy matplotlib pulp pickle tensorflow
   ```

**Project Structure (In Development)**

src/: Will contain the core source code modules.
data/: Place to store sample LP datasets, if used in the future.
notebooks/: For exploratory analysis and model development.
old-workspace/: Current dump ground for old code.

**Limitations and Future Research**

*   **Scope:**  The current model is trained on 2D linear programs with bounded coefficients.
*   **Data Biases:** Further investigation is needed to ensure rigorous debiasing of the training data.

**Future Directions**

*   **Generalization:** Expand the model to handle n-dimensional linear programs.
*   **Optimal Point Estimation:** Explore the potential of predicting optimal LP solutions directly using deep learning.
*   **Algorithmic Integration:**  Investigate how feasibility prediction can be integrated into LP solvers to improve efficiency.
