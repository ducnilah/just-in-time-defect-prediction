*Problem Statement:

Software defects (or software bugs) can have a significant impact on the entire software development lifecycle, affecting many aspects from the reliability and performance of the final product to the efficiency and morale of development teams. Defects that go unnoticed until later stages of development or, worse, after deployment, can lead to costly rework, delayed releases, and even critical system failures. For developers, these issues translate into increased pressure, frustration, and reduced productivity, as they must pay their attention from building new features to tracking down and fixing bugs.  

In the last decade, Just-in-Time (JIT) Defect Prediction has been emerging as a potential solution to address this problem. JIT Defect Prediction focuses on identifying potential defects at the moment they are introduced, during the coding or commit phase, before the code is merged into the main branch. By predicting defects early, this approach enables developers to address issues immediately, preventing them from becoming more complex and costly to fix later. 

*Problem Formulation:
Input: A set of commits C = {c1,c2,c3,...,cn}. Each commit ci contains a set of features fi={fi1,fi2,fi3,...,fin}which are extracted from relevant information of  ci such as commit code changes, authors, etc.
Output: A set of associating probability  P = {p1,p2,p3,...,pn}for input commits, where pi[0, 1] indicate the likelihood that commit ci introduce a defect. 
Task Description

My task is to build a machine learning model to predict pi for ci given  fi. The model will be trained on historical data and will be used to predict new data.
