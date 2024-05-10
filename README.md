Here I provide a description of the files that you can find in this folder.
Please, note that other csv files will be created by running the data analysis file. You need to uncomment the last part of the code, which is commented out, in order to create the files to use in the generalized linear mixed-effects models.
There is also the anonymized quick scan survey, as I did not know where to put it.




COGNITIVE MODELS' RESULTS

Here you find the results of runs of the cognitive models.
The files named "pr0.x" result from running the Double Update Model with lambda equals to 0.x.
The files named axc0 result from running the L2 Model with alpha equals x and cost equals to 0. The value of the cost does not matter, so I could have used any other value.
The file called Baseline Model Results (the txt file) is also needed for the data analysis and shows a complete run of the Baseline Model, with relative learnt concepts.it is.



DATA ANALYSIS 

Run this first. You need to uncomment the last part of the code, which is commented out, in order to create the files to use in the generalized linear mixed-effects models.
One of them requires you to comment out a couple of lines, that made clear in the comments in the file.



EXPERIMENT (experiment.html):

The html file for the experiment was not seen by participants, as they received only a link to the experiment. The link was not provided in the thesis, as I cannot allow for more than 40 runs with my cognition.run account. Thus, running the html file to test the experiment is a better solution, since it is guaranteed to work in the future.

The probability to be assigned to the control group is 1/4. If you specifically want to test the experiment in that group and you do not want to run the experiment until you are assigned to it, you can substitute the content of line 1125 with "listener_procedure_LOT_only".

pwh_label and pw_label are the labels for pragmatic wrong trials discussed in the conclusion.

The last page of the experiment (which thanks participants for their participation) was changed. Therefore, participants may have seen a different page than the one that this code shows. The reason for the change is that someone saw a blank page as the last page, which may have caused confusion (even if the results were successfully saved for everyone).

The way shapes are defined (lines from 26 to 63) was taken from the GitHub forum of jsPsych and slightly modified. The way stimuli are represented (stimulus_listener_1 to 4, which are defined from line 65), was also suggested on that forum. Lastly, the way in which the canvas is divided in two parts (left and right) to accommodate the current stimulus and the record of previous trials is based on suggestions from that forum: the end result greatly differs, but the basic structure is similar.



EXPERIMENTAL RESULTS (exp_results.csv)

This file contains the experimental results. I removed the personal information and useless columns, as for example the version of OS version.



PLOT LEARNING CURVES

This makes the learning curves, run the data analysis file first, in order to create the necessary input files.



RUN COGNITIVE MODELS

This is the implementation of the cognitive models. To use L2 reasoning in pragmatic trials, set L2 to True in line 337. To update both concepts at each trial, set cuncurrent_reasoning to True in line 341. The values of the parameters alpha lambda and also the cost are passed directly to the function run_experiment. Likelihood modifier is set to 0.9 and is defined at the beginning of the code.
