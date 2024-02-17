import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

E_bin_width = 0.05 # photon bin width [keV]
CLOCK_FREQ = 50e06 # MHz 

def _calc_times_in_cycle(df):
    """Calculate and return the event times within each cycle """
    cycle_start_times = df.loc[df.chan == 132, "timestamp"].values
    median_step_duration = np.median(cycle_start_times[1:] - cycle_start_times[0:-1])/CLOCK_FREQ
        
    scan_step_indeces = np.digitize(df.timestamp.values, bins=cycle_start_times)
    time_in_cycle = df["timestamp"].to_numpy(dtype=float)
    all_cycle_start_times = np.zeros_like(time_in_cycle, dtype=float)
    for i in np.unique(scan_step_indeces)[1:]:
        all_cycle_start_times[scan_step_indeces == i] = cycle_start_times[i-1]
    time_in_cycle -= all_cycle_start_times
    time_in_cycle /= CLOCK_FREQ

    df["time_in_cycle"] = time_in_cycle
    df.loc[df.chan != 100, "time_in_cycle"] = -1.0 # flag non-chan100 events
    df.loc[df["time_in_cycle"] > median_step_duration] = -1.0 # flag events outside typical cycle duration

    return df

def plot_time_in_cycle_hist(fname, t_min=0.0, t_max=None, bins=1000):
    """Histogram of event times within each cycle"""
    # Grab and prepare event data 
    df = pd.read_csv(fname, sep="|") 
    df = _calc_times_in_cycle(df)

    # Plot histogram
    df.loc[df["time_in_cycle"] > 0, "time_in_cycle"].hist(bins=bins)
    plt.xlabel("Time within cycle [s]")
    plt.ylabel("Number of events")
    plt.xlim(t_min, t_max)
    plt.show()


def load_event_data(fname, V_DT5=1000, V_cath=-700, file_format="CSV", 
                    calc_times_in_cycle=True):
    """Load data from CSV or HDF file into DataFrame"""
    if file_format.upper() == "CSV":
        df = pd.read_csv(fname, sep="|") 
    elif file_format.upper() in ["HDF", "HDF5"]:
        df = pd.read_hdf(fname)

    # Calibrate X-ray energies ##TODO: Consider handling calibration in mds_sort.py!
    #A = 0.00161422  
    #B = 0.00985710
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
    try:
        E_e = np.unique(df["beam energy"].values[df["ppg_value"].values >= 0]) 
    except KeyError:
        E_e = [None]
    
    spectra = []
    for energy in E_e:
        # sort events into 2D array with one X-ray spectrum for each beam energy
        try:
            t_mask = ((df["time_in_cycle"]>=t_min) & (df["time_in_cycle"]<=t_max))
        except KeyError:
            t_mask = True
        try:
            E_mask = (df["beam energy"] == energy)
            xray_events = df[(df["chan"] == 100) & (df["ppg_value"].values > 0) & t_mask & E_mask]
        except KeyError:
            xray_events = df[(df["chan"] == 100) & (df["ppg_value"].values > 0) & t_mask]
        counts, bin_edges = np.histogram(xray_events["photon energy"].values, 
                                         range=[E_ph_min, E_ph_max], 
                                         bins=N_bins)
        E_ph = bin_edges[0:-1] + bin_edges[1]/2
        spectra.append(counts)

    spectra = np.array(spectra)

    return (E_e, E_ph, spectra)


def get_photon_spectra(fname, V_DT5=1000, V_cath=-700, E_bin_width=E_bin_width, 
                       E_ph_min=0, E_ph_max=20, t_min=0.0, t_max=np.inf, 
                       calc_times_in_cycle=True):
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


def plot_photon_spectra(spectra, E_ph, E_e=None, output_fname=None):
    E_bin_width = E_ph[1] - E_ph[0]
    for i, spec in enumerate(spectra):
        try:
            label = "{} eV".format(E_e[i])
        except TypeError:
            label = None
        plt.plot(E_ph, spec, label=label)
    plt.yscale("log")
    plt.xlabel("Photon energy [keV]")
    plt.ylabel("Counts per {:.3f} keV".format(E_bin_width))
    if E_e is not None:
        plt.legend(title="Beam energy")
    if output_fname is not None:
        plt.savefig(output_fname+".png", dpi=400)
    plt.show()


def fit_photon_peaks(spectra, E_ph, peak_pos, E_ph_min, E_ph_max, sigma0=0.05, 
                     x_var=0.05):
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
    model.set_param_hint("bkg_decay", value=10, min=0)
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


def plot_2D_spectrum(E_ph, E_e, spectra, E_e_min=None, E_e_max=None, E_ph_min=None, 
                     E_ph_max=None, output_fname=None):   
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
    if output_fname:
        plt.savefig(output_fname+".png", dpi=500)
    plt.show()


def plot_E_e_time_evolution(df, E_ph_min=0, E_ph_max=20, 
                            t_min=0, t_max=None, N_time_bins=50, 
                            E_e_min=0, E_e_max=np.inf, plot_every_energy_bin=1):
    """Plot waterfall graph of X-ray intensity vs beam energy"""
    if t_max is None:
        t_max = df["time_in_cycle"].loc[df["time_in_cycle"] > 0].max()
    time_bin_cens = np.linspace(t_min, t_max, num=N_time_bins)
    time_bin_width = time_bin_cens[1] - time_bin_cens[0]
    time_bin_edges = np.append(time_bin_cens - time_bin_width/2, time_bin_cens[-1] + time_bin_width/2)

    E_e = np.unique(df["beam energy"].values[df["ppg_value"].values >= 0])[int(plot_every_energy_bin)::plot_every_energy_bin]
    E_e = E_e[(E_e >= E_e_min) & (E_e <= E_e_max)]
    E_e_bin_width = E_e[1] - E_e[0]
    E_e_edges = np.append(E_e[0] - E_e_bin_width/2, E_e + E_e_bin_width/2) 

    spectra = []
    E_ph_mask = ((df["photon energy"] >= E_ph_min) & (df["photon energy"] <= E_ph_max))
    for t_cen in time_bin_cens: 
        time_mask =  (df["time_in_cycle"] > t_cen - time_bin_width) & (df["time_in_cycle"] <= t_cen + time_bin_width)
        events = df.loc[E_ph_mask & time_mask]
        hist, _ = np.histogram(events["beam energy"].values, bins=E_e_edges)
        spectra.append(hist)
    spectra = np.array(spectra)

    f = plt.figure()
    X, Y = np.meshgrid(E_e_edges, time_bin_edges, indexing='xy')
    plt.pcolor(X, Y, spectra, norm="log")
    plt.xlabel("Electron beam energy [eV]")
    plt.ylabel("Time in cycle [s]")
    c = plt.gca().pcolor(X, Y, spectra)
    cbar = f.colorbar(c, ax=plt.gca())
    cbar.set_label('X-ray events', rotation=90)
    plt.show()
    


def fit_DR_resonance(E_ph, E_e, spectra, peak_pos, E_ph_min, E_ph_max, E_e_min, 
                     E_e_max, amp0=1e05, sigma0=20, slope0=0, vary_slope=False, 
                     min_sigma=10, x_var=30, weighted=True,
                     method="least_squares", reduce_fcn=None):
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
    E_e_bin_width = E_e_fit[1] - E_e_fit[0]
    if weighted:
        weights = 1./summed_spec_cut_errs
    else: 
        weights = None

    # Fit summed spectrum with Gaussians
    from lmfit.models import LinearModel, GaussianModel
    peak_model = GaussianModel
    bkg_model = LinearModel(prefix="bkg_")
    model = bkg_model 
    model.set_param_hint("bkg_slope", value=slope0, vary=vary_slope)
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
    fit_kws = {"reduce_fcn" : reduce_fcn}
    result = model.fit(summed_spec_cut, x=E_e_fit, method=method, 
                       weights=weights, fit_kws=fit_kws)

    # Report fit result and plot
    result.plot(show_init=True, numpoints=1000)
    plt.xlabel("Uncorrected electron beam energy [eV]")
    plt.ylabel("Counts per {:.2f} eV".format(E_e_bin_width))
    plt.show()
    print(result.fit_report())

    return result


def get_beam_energy_from_RR_lines(E_RR, E_bind=4.12067):
    """Calculate corrected beam energy from a fitted RR line
    
    `E_RR` is the centroid of line for radiative recombination into the level 
    with binding energy `E_bind`
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