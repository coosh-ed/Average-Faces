Face Averaging Programme
------------------------

Acknowledgements:
* The project's code is based on the blogpost http://www.learnopencv.com/average-face-opencv-c-python-tutorial.
* The model used to identify the facial features was taken from dlib (dlib.net). This model is in the learning folder.

To run the code, place all images you wish to average over in one folder and run:
   ./main.py  [--calc-diff] path/to/folder

calc-diff is an optional parameter that also outputs a CSV file, "sqr_difference.csv", with the squared differences between the average face and each input image. This may be useful for seeing which faces are shaped most like the average. The plot at https://imgur.com/gallery/8afvY shows the most distinctive and "average" shaped faces for the male TDs.

Warning: The programme transforms the images into usable versions and places them in a folder called "[FolderName]Transformed[TimeStamp]". Feel free to delete this afterwards.

Also included in the helpers folder is a script that scrapes the web for pictures of the current TDs in Irish Government. When run, it saveds pictures of the male and female TDs in folders maleTDs and femaleTDs respectively. 

The files "MaleTDs.jpg" and "FemaleTDs.jpg" show the average faces for the most recent (2017) members of Dail Eireann (Irish Parliament).
