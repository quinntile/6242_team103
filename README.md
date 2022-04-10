# 6242_team103
Proj folder for team103. 
## Common git commands that are useful for our project:
1) git init --> initialize a git repo in your local folder, so that codes and data can be cloned into.
2) git config --username, git config --email --> config your local repo to let it know that you own the repo and connect to our remote repo.
3) git clone --> once you init'd a repo on your local machine, you can use git clone https://github.com/quinntile/6242_team103 to clone this repo to your local folder,                   please note, this is a private repo, so there maybe error saying "no such repo" or can't find repo, first make sure you are on the colloborator list,                      then use your github account token to clone it to your folder. Refer to https://linuxpip.org/clone-private-repo-github/ if you need help.
4) git add --> once you modified some codes, or added data, only add the specific file/modification to the staging area(for example, git add /6242_team103/data/cleaning unemployment.ipynb), DO NOT use add . (add all modifications), as this may cause a lot of conflicts.
5) git commit -m "your message", commit your changes to the local branch
6) git pull --> get updated git repo, for example, our team member's contribution needs to be updated to your local folder before you push
7) git push --> push your modifications to the github repo, so that other memebers can pull and see.


## For time series prediction of features, only one zip code from seattle has been done, with a monthly frequency and predict into 6 month into the future. Once we are fixed with which set of features to choose, we can run time series prediction on them simultaneously
