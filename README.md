# Ava Downey Undergraduate Honors Thesis

As an increasing number of Artificial Intelligence (AI) systems are ingrained in our day-to-day lives, it is crucial that they are fair and trustworthy. Unfortunately, this is often not the case for predictive policing systems, where there is evidence of bias towards age as well as race and sex leading to many people being mistakenly labeled as likely to be involved in a crime. In a system that already is under criticism for its unjust treatment of minority groups, it is crucial to find ways to mitigate this negative trend. In this work, we explored and evaluated the infusion of domain knowledge in the predictive policing system to minimize the prevailing fairness issues. The experimental results demonstrate an increase in fairness across all of the metrics for all of the protected classes bringing more trust into the predictive policing system by reducing the unfair policing of people.

This repository provides the code used to come to the conclusions found in this thesis.

## How to Use

1. Install all required python packages to local environment using `pip install -r requirements.txt`. If problems arise with AIF360 installation, try installing with conda instead.
1. Pull repo down into local environment.
1. Fairness models ran are all under the [models](https://github.com/avasdowney/predictive_policing_thesis/tree/master/models) folder, and are seperated by the domain knowledge, or lack thereof included in the model. Within these subfolders, run the `[folder]_census_fairness.py` file to see fairness metrics. The classification metrics are calculated within the `[folder]_census_models.py` files.

## Dataset

The dataset used for this project can be found [here](https://data.cityofchicago.org/Public-Safety/Strategic-Subject-List-Historical/4aki-r3np).

The data dictionary for this dataset can be found [here](https://www.opendatanetwork.com/dataset/data.cityofchicago.org/4aki-r3np).


## Fairness Tools

IBM [AIF360](https://github.com/Trusted-AI/AIF360/tree/master) is being used to calculated bias and fairness. Their reweight tool is also used to mitigate bias.

## Resources Used

I used several tutorials to help create the code to test my hypotheses. They are listed below.

1. https://github.com/bryantruong/examingBiasInAI
1. https://consileon.ai/wp-content/uploads/2021/08/Fairness-in-AI-Systems.pdf 
1. https://colab.research.google.com/drive/1QVXhGr93ex9Gdovo7WO3ogK39Qn8uQ7Q?usp=sharing#scrollTo=BojAx75gCaKl
