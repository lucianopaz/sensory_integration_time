import leakyIntegrator, os
from leakyIntegrator import AllData, Stimulator, Leaky, Fitter, Fitter_plot_handler, Fitter_ploter

if __name__=="__main__":
    options = leakyIntegrator.fits_module.parse_input(script_name = os.path.basename(__file__))
    leakyIntegrator.fits_module.main(**options)
