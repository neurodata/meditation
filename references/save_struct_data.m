proj1 = Data.compassion(1).projX;
proj2 = Data.compassion(2).projX;

csvwrite('C:\\Users\\Ronan Perry\\Documents\\JHU\\jovo-lab\\meditation\\data\\matlab_compassion_025_thresh300.csv',proj1)
csvwrite('C:\\Users\\Ronan Perry\\Documents\\JHU\\jovo-lab\\meditation\\data\\matlab_compassion_050_thresh300.csv',proj2)

csvwrite('C:\\Users\\Ronan Perry\\Documents\\JHU\\jovo-lab\\meditation\\data\\matlab_inv0.csv',Data.compassion(1).inv);
csvwrite('C:\\Users\\Ronan Perry\\Documents\\JHU\\jovo-lab\\meditation\\data\\matlab_V0.csv',Data.compassion(1).V);