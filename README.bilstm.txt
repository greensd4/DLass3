########################### Authors ##################################

Daniel Greenspan, 308243948
Eilon Bashari, 308576933

#######################################################################

########################## How To Run #################################

First step for training on your data:
    To run the bilstmTrain.py for training on your data please insert
    the command:
        python bilstmTrain.py <repr> <train file> <model file> -d <dev file> -t <tagger type>
        Arguments:
            <repr> is a b c or d as specified in the assignment.
            <train file> the file which the model and the neural network will be trained on.
            <model file> is the output file where the neural network data and the model data will be stored.
        Options:
            -t TYPE, --type=TYPE  choose POS/NER tagging (pos/ner) - REQUIRED
            -d DEV, --dev=DEV     dev file name
    Please notice:
        *that the -t flag is required and necessary for this program
        *that if dev file is not entered in -d flag we will use our own dev file.
        *that the output of this program is 2 files one for the model and one for
            the data with the suffix _data to the model name

Second step for testing your data:
    To run the bilstmPredict.py for testing on different data please insert
    the command:
        python bilstmPredict.py <repr> <model file> <input file> -t <type>
        Arguments:
            <repr> is a b c or d as specified in the assignment.
            <model file> is the input file where the model reads its parameters from.
            <input file> this is the test file which we do our predictions on.
        Options:
            -t TYPE, --type=TYPE  choose POS/NER tagging (pos/ner) - REQUIRED
   Please notice:
       *that the -t flag is required and necessary for this program

#######################################################################


########################## Packages ####################################

                DyNet , NumPy , Json, MatplotLib

#######################################################################