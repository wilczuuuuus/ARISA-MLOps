# ARISA-MLOps
## Sessions 01 and 02
### Prerequisites  
Install python 3.11 with py manager on your local machine.  
Install Visual Studio Code on your local machine.  
Create a kaggle account and create a kaggle api key kaggle.json file.  
Move the kaggle.json to   
C:\Users\USERNAME\\.kaggle folder for windows,  
/home/username/.config/kaggle folder for mac or linux.  
### Local
Fork repo to your own github account.  
Clone forked repo to your local machine.  
Open VS Code and open the repo directory.  
In the VS Code terminal run the following to create a new python virtual environment:  
```
py -3.11 -m venv .venv
```
windows
```
.\.venv\Scripts\activate
```
mac or linux  
```
source .venv/bin/activate
```
and then open up notebook 02 and attempt to run the cells.  

### Github Codespaces  
Fork repo to your own github account.  
Click the code button, select codespaces tab click create codespace on main.  
Optionally, after the codespace has finished loading click the blue button on the bottom left and select open in visual studio code.  
Make sure to turn off the codepace when done, you have a limited amount of free runtime hours.  

## Sessions 03 and 04
The focus of code introduced in sessions 03 and 04 (March 8th and 9th) is moving from the codebase developed in sessions 01 and 02,  
namely moving from a codebase prepared for MLOps, to actually doing reproducible ML.  
We will be implementing the experimentation and train/predict pipelines in the below diagram:
![image](https://github.com/user-attachments/assets/f11539b6-9bcc-4a04-b89f-97e6e7383bf2)
(source https://ml-ops.org/content/mlops-principles)
More specifically, we will be hosting a MLflow tracking server in a Codespace (but AWS EC2 could also be an option),
and running the two pipelines in GitHub Actions Workflows, see architecture below:
![arisa-githut-architecture](https://github.com/user-attachments/assets/fb13f63b-e821-4b9b-869a-e3f2ea431a9b)
![arisa resolve](https://github.com/user-attachments/assets/76eedd72-0326-4d80-879c-9f6761349032)
### Getting the architecture up and running
WIP
