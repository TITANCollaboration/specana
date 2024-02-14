import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

##V_DT5 = 1000 # for run #01525 # 1000
##V_cath = -700 #-1800 # for run #01525 #  -700 

E_bin_width = 0.05 # photon bin width [keV]
CLOCK_FREQ = 50e06 # MHz 

def _calc_times_in_cycle(df):
    """Calculate and return the event times within each cycle """
    cycle_start_times = df.loc[df.chan == 132, "timestamp"].values
    median_step_duration = np.median(cycle_start_times[1:] - cycle_start_times[0:-1])/CLOCK_FREQ
    df["scan_step"] = np.digitize(df.timestamp.values, bins=cycle_start_times)
    time_in_cycle = []
    for idx in set(df["scan_step"]): 
        time_chunk = df.loc[df.scan_step == idx]["timestamp"]
        cycle_time = (time_chunk - cycle_start_times[idx-1])/CLOCK_FREQ
        time_in_cycle.extend(cycle_time)
    df["time_in_cycle"] = time_in_cycle
    df.loc[df.chan != 100, "time_in_cycle"] = -1.0 # handle non-ULGe events
    df.loc[df["time_in_cycle"] > median_step_duration] = -1.0 # handle events outside typical cycle duration

    return df

def plot_time_in_cycle_hist(fname, t_min=0.0, t_max=np.inf, bins=1000):

    # Grab and prepare event data 
    df = pd.read_csv(fname, sep="|") 
    df = _calc_times_in_cycle(df, t_min=t_min, t_max=t_max)

    # Plot histogram
    df.loc[df["time_in_cycle"] > 0, "time_in_cycle"].hist(bins=bins)
    plt.xlabel("Time within cycle [s]")
    plt.ylabel("Number of events")
    plt.show()


def load_event_data(fname, V_DT5=1000, V_cath=-700, calc_times_in_cycle=True):
    """Load data from root file into DataFrame"""
    df = pd.read_csv(fname, sep="|") 
    #display(df.head(20))

    #TODO Convert timestamps to datetimes
    #df["datetimes"] = datetime.fromtimestamp(df["timestamp"])

    # Calibrate X-ray energies
    #A = 0.0149201298 * 0.1175 ###### GET NEW CALIBRATION!
    #B = -0.271381276
    # John's results:
    A = 0.001614397
    B = 0.017175819
    E = A*df.pulse_height + B
    df["photon energy"] = E
    
    if calc_times_in_cycle:
        # Add column with event times relative to last mdpp trigger
        df["time_in_cycle"] = -1.0
        df = _calc_times_in_cycle(df)

    # Add column with uncorrected beam energies
    try:
        df["beam energy"] = V_DT5 + df["ppg_value"] + np.abs(V_cath) 
        df.loc[df["ppg_value"] < 0, "beam energy"] = 0 # handle columns with ppg_output == -1
    except KeyError:
        pass

    return df
    
def _get_spectra(df, E_bin_width=E_bin_width, E_ph_min=0, E_ph_max=20, 
                 t_min=0.0, t_max=np.inf):

    N_bins = int( (E_ph_max - E_ph_min)/ E_bin_width )

    # Get beam energies
    ##V_bias = np.unique(df["ppg_value"].values[df["ppg_value"].values >= 0])
    try:
        E_e = np.unique(df["beam energy"].values[df["ppg_value"].values >= 0]) 
    except KeyError:
        E_e = [None]
    
    spectra = []
    for energy in E_e:
        # sort events into 2D array with one X-ray spectrum for each beam energy
        t_mask = ((df["time_in_cycle"]>=t_min) & (df["time_in_cycle"]<=t_max))
        try:
            E_mask = (df["beam energy"] == energy)
            xray_events = df[ (df["chan"] == 100) & (df["ppg_value"].values > 0) & t_mask & E_mask]
        except KeyError:
            xray_events = df[ (df["chan"] == 100) & t_mask]
        counts, bin_edges = np.histogram(xray_events["photon energy"].values, 
                                         range=[E_ph_min, E_ph_max], 
                                         bins=N_bins)
        E_ph = bin_edges[0:-1] + bin_edges[1]/2
        spectra.append(counts)

    spectra = np.array(spectra)

    return (E_e, E_ph, spectra)


def get_photon_spectra(fname, V_DT5=1000, V_cath=-700, E_bin_width=E_bin_width, 
                       E_ph_min=0, E_ph_max=20, t_min=0.0, t_max=np.inf):
    """Create a binned photon spectrum 
    
    The `t_min` and `t_max` arguments can be used to gate on times of interest 
    within a cycle. 

    """
    df = load_event_data(fname, V_DT5=V_DT5, V_cath=-V_cath, 
                         calc_times_in_cycle=calc_times_in_cycle)

    (E_e, E_ph, spectra) = _get_spectra(df, E_bin_width=E_bin_width, 
                                       E_ph_min=E_ph_min, E_ph_max=E_ph_max, 
                                       t_min=t_min, t_max=t_max)
    
    return (E_e, E_ph, spectra)



def fit_photon_peaks(spectra, E_ph, peak_pos, E_ph_min, E_ph_max, sigma0=0.05, x_var=0.05):
    """Fit X-ray spectrum with Gaussian"""
    # Plot spectrum with peak markers
    E_ph_mask = np.logical_and(E_ph >= E_ph_min, E_ph <= E_ph_max)
    E_ph_fit = E_ph[E_ph_mask]
    photon_counts = spectra.flatten()[E_ph_mask]
    photon_count_errs = np.maximum(np.sqrt(photon_counts), 1)
    for E in peak_pos:
        plt.gca().axvline(E, linestyle="dashed")
    plt.plot(E_ph_fit, photon_counts, ".-")
    plt.xlim(E_ph_min, E_ph_max)
    plt.show()


    # Build fit model
    from lmfit.models import ExponentialModel, GaussianModel
    peak_model = GaussianModel
    bkg_model = ExponentialModel(prefix="bkg_")
    model = bkg_model 
    model.set_param_hint("bkg_amplitude", value=10)
    model.set_param_hint("bkg_decay", value=1, min=0)
    for i, x_pos in enumerate(peak_pos):
        pref = "p{}_".format(i)
        model += peak_model(prefix=pref)
        model.set_param_hint(pref+"amplitude", value=10, min=0)
        model.set_param_hint(pref+"center", value=x_pos, min=x_pos - x_var, max=x_pos + x_var)
        model.set_param_hint(pref+"sigma", value=sigma0, min=0)
        if i > 0:
            model.set_param_hint(pref+"sigma", expr="p0_sigma")

    # Fit 
    result = model.fit(photon_counts, x=E_ph_fit, weights=1/photon_count_errs)

    # Report fit result and plot
    result.plot(show_init=True, numpoints=1000)
    plt.show()
    print(result.fit_report())
    
    return result


def get_beam_energy_from_RR_lines(E_RR, E_bind=4.12067): 
    E_e_corr = E_RR - E_bind
    return E_e_corr 

def correct_beam_energy(E, E_ref_exp, E_ref_lit, slope_corr=0.0):
    delta_E_ref = E_ref_exp - E_ref_lit
    if E == 0:
        E_corr = 0
    else:
        delta_E = delta_E_ref*np.sqrt(E_ref_exp/E) 
        E_corr = E - delta_E + slope_corr*(E-E_ref_exp) # last term for linear space charge correction
    return E_corr
correct_beam_energy = np.vectorize(correct_beam_energy)


def plot_2D_spectrum(E_ph, E_e, spectra, E_e_min=None, E_e_max=None, E_ph_min=None, 
                     E_ph_max=None, run_number= None, save_fig=True):   
    """Plot 2D plot of raw X-ray data"""
    # Prepare data for 2D plot
    E_ph_bin_width = E_ph[1] - E_ph[0]
    E_ph_edges = np.append(E_ph[0] - E_ph_bin_width/2, E_ph + E_ph_bin_width/2) 
    E_e_bin_width = E_e[1] - E_e[0]
    E_e_edges = np.append(E_e[0] - E_e_bin_width/2, E_e + E_e_bin_width/2) 
    X, Y = np.meshgrid(E_ph_edges, E_e_edges, indexing='xy')

    # Plot
    plt.figure(figsize=(8,6))
    plt.title("Run #{} - Ar DRR".format(run_number))
    plt.pcolor(X, Y, spectra, norm="log")
    plt.xlim(E_ph_min, E_ph_max)
    plt.ylim(E_e_min, E_e_max)
    plt.xlabel("Photon energy [keV]")
    plt.ylabel("Uncorrected electron beam energy [eV]")
    if save_fig:
        if run_number is not None:
            plt.savefig("run{}_Ar_DRR_2D_scan".format(run_number), dpi=500)
        else: 
            plt.savefig("run{}_Ar_DRR_2D_scan".format(run_number), dpi=500)
    plt.show()


def plot_E_e_time_evolution(df, E_ph_min=0, E_ph_max=20, t_min=0, t_max=None, N_time_bins=50, plot_every_energy_bin=1):
    if t_max is None:
        t_max = df["time_in_cycle"].loc[df["time_in_cycle"] > 0].max()
    time_bin_cens = np.linspace(t_min, t_max, num=N_time_bins)
    time_bin_width = time_bin_cens[1] - time_bin_cens[0]
    time_bin_edges = np.append(time_bin_cens - time_bin_width/2, time_bin_cens[-1] + time_bin_width/2)

    E_e = np.unique(df["beam energy"].values[df["ppg_value"].values >= 0])[int(plot_every_energy_bin)::plot_every_energy_bin]
    E_e_bin_width = E_e[1] - E_e[0]
    E_e_edges = np.append(E_e[0] - E_e_bin_width/2, E_e + E_e_bin_width/2) 

    spectra = []
    E_ph_mask = ((df["photon energy"] > E_ph_min) & (df["photon energy"] < E_ph_max))
    for t_cen in time_bin_cens: 
        time_mask =  (df["time_in_cycle"] > t_cen - time_bin_width) & (df["time_in_cycle"] <= t_cen + time_bin_width)
        events = df.loc[E_ph_mask & time_mask]
        hist, _ = np.histogram(events["beam energy"].values, bins=E_e_edges)
        spectra.append(hist)
    spectra = np.array(spectra)

    plt.plot(spectra.transpose())
    plt.show()

    f = plt.figure()
    X, Y = np.meshgrid(E_e_edges, time_bin_edges, indexing='xy')
    plt.pcolor(X, Y, spectra, norm="log")
    plt.xlabel("Electron beam energy [eV]")
    plt.ylabel("Time in cycle [s]")
    c = plt.gca().pcolor(X, Y, spectra)
    cbar = f.colorbar(c, ax=plt.gca())
    cbar.set_label('X-ray events', rotation=270)
    plt.show()
    


def fit_DR_resonance(spectra, peak_pos, E_ph_min, E_ph_max, E_e_min, E_e_max, 
                     amp0=1e05, sigma0=20, min_sigma=10, x_var=30):
    """Fit DR resonance to calibrate electron beam energy [eV]"""
    
    # Filter out photon energies of interest and create summed spectrum
    E_ph_mask = np.logical_and(E_ph >= E_ph_min, E_ph <= E_ph_max)
    summed_spec = np.sum(spectra.transpose()[E_ph_mask], axis=0)
    for E in peak_pos:
        plt.gca().axvline(E, linestyle="dashed")
    plt.plot(E_e, summed_spec, ".-")
    plt.xlim(E_e_min, E_e_max)
    plt.show()

    # Cut summed spectrum to energy range to fit
    E_e_mask = np.logical_and(E_e >= E_e_min, E_e <= E_e_max)
    summed_spec_cut = summed_spec[E_e_mask]
    summed_spec_cut_errs =  np.maximum(np.sqrt(summed_spec_cut), 1.) 
    E_e_fit = E_e[E_e_mask]

    # Fit summed spectrum with Gaussians
    from lmfit.models import LinearModel, GaussianModel
    peak_model = GaussianModel
    bkg_model = LinearModel(prefix="bkg_")
    model = bkg_model 
    model.set_param_hint("bkg_slope", value=-5)
    model.set_param_hint("bkg_intercept", value=10000, min=0)
    for i, x_pos in enumerate(peak_pos):
        pref = "p{}_".format(i)
        model += peak_model(prefix=pref)
        model.set_param_hint(pref+"amplitude", value=amp0, min=0)
        model.set_param_hint(pref+"center", value=x_pos, min=x_pos - x_var, max=x_pos + x_var)
        model.set_param_hint(pref+"sigma", value=sigma0, min=min_sigma)
        if i > 0:
            model.set_param_hint(pref+"sigma", expr="p0_sigma")

    # Fit
    result = model.fit(summed_spec_cut, x=E_e_fit, weights=summed_spec_cut_errs)

    # Report fit result and plot
    result.plot(show_init=True, numpoints=1000)
    plt.show()
    print(result.fit_report())

    return result


def get_beam_energy_from_RR_lines(E_RR, E_bind=4.12067):
    """Calculate corrected beam energy from a fitted RR line
    
    Only applicable for a fixed energy.
    """ 
    E_e_corr = E_RR - E_bind
    return E_e_corr 


def correct_beam_energy(E, E_ref_exp, E_ref_lit, slope_corr=0.0):
    """Correct e-beam energy using a DR resonance and apply optional slope correction

    Applicable over a range of e-beam energies (but a slope correction might be 
    required to match exp. data)

    """
    delta_E_ref = E_ref_exp - E_ref_lit
    if E == 0:
        E_corr = 0
    else:
        delta_E = delta_E_ref*np.sqrt(E_ref_exp/E) 
        E_corr = E - delta_E + slope_corr*(E-E_ref_exp) # last term for linear space charge correction
    return E_corr
correct_beam_energy = np.vectorize(correct_beam_energy)