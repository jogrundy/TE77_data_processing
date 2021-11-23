# TE77_data_processing

This is a tutorial in the use of the scripts I have written to take an hpf file from the TE77 reciprocating tribometer, read the file in to an Hpf object using the 'read_hpf' file, then put that data in to bins, so that the data from a particular part of the surface can be examined. 

These bins are n strokes of the roller by p bins matrices, and are written to file in a 'bins/' directory, ideally in the same place as you keep your hpf files. This saves much time in binning the data, as it can be read. The files have each stroke indexed by its start time, with the column labeled by the displacement. 

The files are read to give a set of DataFrames, allowing easy access and data manipulation. Each data frame is for a particular direction and data type, usualy 'ES - left', 'ES - right', 'Friction - left', 'Friction - right'.  The data frame has the stroke start time as the index, and the columns are the displacements. The displacements are the lower end of the bins, and are different for ES and Friction as they cover different displacement ranges. 

The use of my outlier detection library (odds) and some other visualisation code is demonstrated, all in the TE77_data.pynb file.
