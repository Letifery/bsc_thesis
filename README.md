This project encompasses the source code of my bachelor thesis

Important things to mention: 
- The recall/precision for binaryclassification is wrong when the program is executed and has to be calculated manually from the saved torch datapoints
- Most Models (such as the vision transformer model on REAL-ESRGAN SR'ed images of the CRTX dataset) couldn't be uploaded due to their large size


Tools: Contains various tools concerning image pre-/postprocessing, not directly connected to the main loop of the program
visuals: Output folder for visuals generated
data: Contains various open-source datasets and samples from the CRTX dataset referenced in the thesis
models: Saves the savestates of the best/worst model and stats over the folds used
Real-ESRGAN: Fork of Real-ESRGAN to use SR on images (https://github.com/xinntao/Real-ESRGAN
framework: Various tools used to process the used data/model
