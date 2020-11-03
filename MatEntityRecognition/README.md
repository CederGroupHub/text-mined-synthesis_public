# Synthesis Materials Recognizer (SMR) for Materials Entity Recognition (MER)

Extract materials from a paragraph, and recognize the targets and precursors in those materials  

Install:  

	Then
		git clone git@github.com:CederGroupHub/MatEntityRecognition.git 
		cd MatEntityRecognition
		pip install -e .
	
	Spacy is used. If there is an error saying: 
	    "Can't find model 'en'..." 
	It is because the spacy data is not downloaded. Please use:
	    python -m spacy download en
	    
    MaterialParser is used. Please find it here:
        https://github.com/CederGroupHub/MaterialParser

Use:

	from materials_entity_recognition import MatRecognition   
	model = MatRecognition()  
	result = model.mat_recognize(input_paras)  

Parameters:

	Input: list of plain text of paragraphs or plian text of a paragraph. 
	Note: input a list of paragraphs (recommended) is much faster than inputting them one by one in a loop!  
	Output: a list of (list of) dict objects, containing all materials, precursors, targets, and other materials for each sentence in the input paragraphs.  
