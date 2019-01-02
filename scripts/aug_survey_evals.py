import scda # https://github.com/neilzim/SCDA
import pprint
import logging
import os
import numpy as np
import pprint
import shutil
import glob

scda.configure_log()

#////////////////////////////////////////////////////////////////////
# Set the star angular diameters to evaluate, in units of lambda0/D
#////////////////////////////////////////////////////////////////////
star_diam_vec = np.concatenate([np.linspace(0,0.09,10), np.linspace(0.1, 1, 10), np.array([2., 3., 4.])])
#star_diam_vec = [0.05, 0.10]
print("Star diameters: {:}".format(star_diam_vec))

#////////////////////////////////////////////////////////////////////
# Load the design survey archive
#////////////////////////////////////////////////////////////////////
aug04survey_cusp = scda.load_design_param_survey('aug_survey04_cusp.pkl')

#////////////////////////////////////////////////////////////////////
# List of coronagraph designs to evaluate
#////////////////////////////////////////////////////////////////////
coron_list = [aug04survey_cusp.coron_list[2],
              aug04survey_cusp.coron_list[10],
              aug04survey_cusp.coron_list[12],
              aug04survey_cusp.coron_list[31],
              aug04survey_cusp.coron_list[36],
              aug04survey_cusp.coron_list[48],
              aug04survey_cusp.coron_list[54],
              aug04survey_cusp.coron_list[67],
              aug04survey_cusp.coron_list[79],
              aug04survey_cusp.coron_list[91],
              aug04survey_cusp.coron_list[104],
              aug04survey_cusp.coron_list[120],
              aug04survey_cusp.coron_list[125],
              aug04survey_cusp.coron_list[143],
              aug04survey_cusp.coron_list[157],
              aug04survey_cusp.coron_list[171]]

for coron in coron_list:
    coron.fileorg['eval dir'] = os.path.join(coron.fileorg['work dir'], 'evals_hires')
    telap_flag = coron.get_metrics()
#    print("telap_flag = {}".format(telap_flag))

    #//// LOW RES //////
#    coron.write_eval_products(star_diam_vec=star_diam_vec, pixscale_lamoD=0.5, Nlam=3, 
#                              norm='aperture', second_curve_diam=0.2, dpi=600)

    #//// HIGH RES /////
    coron.write_eval_products(star_diam_vec=star_diam_vec, pixscale_lamoD=0.25, Nlam=7, 
                              norm='aperture', second_curve_diam=0.2, dpi=600)
