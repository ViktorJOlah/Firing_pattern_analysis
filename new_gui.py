#%%
from dearpygui.core import get_value
import numpy as np
from numpy.core.fromnumeric import argmax
import pyabf
from scipy import *

from dearpygui.core import *
from dearpygui.simple import *

#%%

abf_name = "vmi.abf"

def open_myabf(abf_name):
    return pyabf.ABF(abf_name)


abf = open_myabf(abf_name)
abf.setSweep(2,0)
#%%

#Params
file1 = 0
abf.setSweep(1,1)

dt = abf.sweepX[1] - abf.sweepX[0]

start1 = 0.63911
lenght1 = 1

#start1 = ent2.get()
#lenght1 = ent3.get()

stim_start = int(start1 / dt)
stim_end = int(stim_start + (lenght1 / dt))


#abf.channelCount = 4
#abf.sweepCount = 11

#%%

def set1():
    global AP_peak
    global AP_threshold
    global AP_hwdt
    global AHP
    global dvdt_max
    global membrane_tau
    global membrane_capacitance
    global Input_resistance
    global resting_membrane_potential
    global sag_amplitude
    global sag_timing
    global mean_firing_frequency
    global max_firing_frequency
    global rheobase
    global accommodation_ratio
    global frequency_list
    global current_list


    AP_peak = 'nan'
    AP_threshold = 'nan'
    AP_hwdt = 'nan'
    AHP = 'nan'
    dvdt_max = 'nan'
    membrane_tau = 'nan'
    membrane_capacitance = 'nan'
    Input_resistance = 'nan'
    resting_membrane_potential = 'nan'
    sag_amplitude = 'nan'
    sag_timing = 'nan'
    mean_firing_frequency = 'nan'
    max_firing_frequency = 'nan'
    rheobase = 'nan'
    accommodation_ratio = 'nan'
    frequency_list = 'nan'
    current_list = 'nan'

set1()

file1 = 'temp.txt'
def write():
    with open(file1, "w") as myfile:
        myfile.write("AP peak: "+ "\t" + "\t" + "\t" + "\t" + str(AP_peak) + "\n")
        myfile.write("AP threshold: "+ "\t" + "\t" + "\t" + "\t" + str(AP_threshold) + "\n")
        myfile.write("AP hwdt: "+ "\t" + "\t" + "\t" + "\t" + str(AP_hwdt) + "\n")
        myfile.write("AHP: "+ "\t" + "\t" + "\t" + "\t" + str(AHP) + "\n")
        myfile.write("dV/dt maximum: "+ "\t" + "\t" + "\t" + "\t" + str(dvdt_max) + "\n")
        myfile.write("membrane tau: "+ "\t" + "\t" + "\t" + "\t" + str(membrane_tau) + "\n")
        myfile.write("membrane capacitance: "+ "\t" + "\t" + "\t" + "\t" + str(membrane_capacitance) + "\n")
        myfile.write("input resistance: "+ "\t" + "\t" + "\t" + "\t" + str(Input_resistance) + "\n")
        myfile.write("resting membrane potential: "+ "\t" + "\t" + "\t" + "\t" + str(resting_membrane_potential) + "\n")
        myfile.write("sag amplitude: "+ "\t" + "\t" + "\t" + "\t" + str(sag_amplitude) + "\n")
        myfile.write("sag timing: "+ "\t" + "\t" + "\t" + "\t" + str(sag_timing) + "\n")
        myfile.write("mean firing frequency: "+ "\t" + "\t" + "\t" + "\t" + str(mean_firing_frequency) + "\n")
        myfile.write("maximum firing frequency: "+ "\t" + "\t" + "\t" + "\t" + str(max_firing_frequency) + "\n")
        myfile.write("rheobase: "+ "\t" + "\t" + "\t" + "\t" + str(rheobase) + "\n")
        myfile.write("accommodation ratio: "+ "\t" + "\t" + "\t" + "\t" + str(accommodation_ratio) + "\n")
        if frequency_list is list:
            myfile.write("firing rate per current density: " + "\n")
            for i in range(len(current_list)):
                myfile.write(str(current_list[i]) + "\t" + str(frequency_list[i]) + "\n")


def write():
    with open(file1, "w") as myfile:
        myfile.write(str(AP_peak) + "\n")
        myfile.write(str(AP_threshold) + "\n")
        myfile.write(str(AP_hwdt) + "\n")
        myfile.write(str(AHP) + "\n")
        myfile.write(str(dvdt_max) + "\n")
        myfile.write(str(membrane_tau) + "\n")
        myfile.write(str(membrane_capacitance) + "\n")
        myfile.write(str(Input_resistance) + "\n")
        myfile.write(str(resting_membrane_potential) + "\n")
        myfile.write(str(sag_amplitude) + "\n")
        myfile.write(str(sag_timing) + "\n")
        myfile.write(str(mean_firing_frequency) + "\n")
        myfile.write(str(max_firing_frequency) + "\n")
        myfile.write(str(rheobase) + "\n")
        myfile.write(str(accommodation_ratio) + "\n")
        if frequency_list is list:
            myfile.write("firing rate per current density: " + "\n")
            for i in range(len(current_list)):
                myfile.write(str(current_list[i]) + "\t" + str(frequency_list[i]) + "\n")

write()

#%%

def calc_rheobase():

    global rheobase

    abf.setSweep(0, int(get_value("Voltage channel")))
    dt = abf.sweepX[1] - abf.sweepX[0]
    start1 = get_value("Stimulus start (s)")
    lenght1 = get_value("Stimulus lenght (s)")

    stim_start = int(float(start1) / dt)
    stim_end = int(stim_start + (float(lenght1) / dt))

    sweep = 0
    for i in range(abf.sweepCount):
        sweep = i
        abf.setSweep(i, int(get_value("Voltage channel")))
        if np.max(abf.sweepY[:]) > -20:
            break
    
    abf.setSweep(sweep, int(get_value("Command channel")))
    rheobase1 = np.mean(abf.sweepY[stim_start : (stim_start + 100)])

    rheobase = rheobase1

def calc_spike_params():

    global AP_peak
    global AP_hwdt
    global AHP
    global AP_threshold
    global dvdt_max

    abf.setSweep(0, int(get_value("Voltage channel")))
    dt = abf.sweepX[1] - abf.sweepX[0]
    start1 = get_value("Stimulus start (s)")
    lenght1 = get_value("Stimulus lenght (s)")

    stim_start = int(float(start1) / dt)
    stim_end = int(stim_start + (float(lenght1) / dt))

    abf.setSweep(int(get_value("Sweep for spike parameters")), int(get_value("Voltage channel")))
    threshat = 0
    half2 = 0
    for i in range(int(float(lenght1) / dt)):
        if abf.sweepY[stim_start + i] > -20:
            st1 = int(stim_start + i)                               #calc peak
            st2 = int(stim_start + i + (0.002 / dt))
            peak = np.max(abf.sweepY[st1 : st2])
            peakat = np.argmax(abf.sweepY[st1 : st2]) + st1           

            dvdt = np.gradient(abf.sweepY, dt * 1000)                 #calculate dvdt_max
            st1 = int(stim_start + i - (0.002 / dt))
            st2 = int(stim_start + i + (0.002 / dt))
            dvdt_max1 = np.max(dvdt[st1 : st2])

            st1 = int(stim_start + i)                               #calculate threshold
            for k in range(int(0.002 / dt)):
                if dvdt[st1 - k] < 50:
                    thresh = abf.sweepY[st1 - k]
                    threshat = st1 - k
                    break

            st1 = int(stim_start + i) 
            st2 = int(stim_start + i + (0.006 / dt))                #calc AHP
            AHP1 = abs(np.min(abf.sweepY[st1 : st2]) - thresh)
            
            half1 = (peakat - threshat) / 2 + threshat
            st1 = peakat
            st2 = peakat + int(0.002 / dt)
            for i in range(st1,st2):
                if abf.sweepY[i] < abf.sweepY[int(half1)]:
                    half2 = i
                    break
            hwdt = (half2 - half1) * dt *1000

            break
        
    calc_rheobase()

    AP_peak = peak
    AP_hwdt = hwdt
    AHP = AHP1
    AP_threshold = thresh
    dvdt_max = dvdt_max1
    




# %%



def calc_resting():

    global resting_membrane_potential
    global Input_resistance

    abf.setSweep(0, int(get_value("Voltage channel")))
    dt = abf.sweepX[1] - abf.sweepX[0]

    stim_start = int(float(get_value("Passive stimuli start (s)")) / dt)
    stim_end = int(stim_start + (float(get_value("Passive stimuli lenght (s)")) / dt))

    abf.setSweep(int(get_value("Sweep for passive parameters")), int(get_value("Voltage channel")))

    mean1 = np.mean(abf.sweepY[0 : stim_start])                             #calculate Rm
    mean2 = np.mean(abf.sweepY[int(stim_start + (0.1 / dt)) : stim_end])
    abf.setSweep(int(get_value("Sweep for passive parameters")), int(get_value("Command channel")))
    holding = np.mean(abf.sweepY[stim_start - 1000 : stim_start-10])
    pas_stim = np.mean(abf.sweepY[stim_start + 10 : stim_start + 110]) - holding

    Rm = (abs(mean1-mean2) / abs(pas_stim) ) * 1000
    
    resting = mean1 - (Rm * holding) / 1000

    print(mean1, mean2, holding, pas_stim, Rm, resting)
    Input_resistance = Rm
    resting_membrane_potential = resting

def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b

p0 = (10, 1, 10)

def calc_pas_params():

    vmi = [[] for i in range(abf.sweepCount)]
    for sweep in range(abf.sweepCount):
        abf.setSweep(sweep,int(get_value("Voltage channel")))
        vmi[sweep] = np.asarray(abf.sweepY)

    average_sweep = np.average(vmi, axis=0)

    print(f"average_lenght: {average_sweep.shape}")
    #try:
    import scipy.optimize
    global membrane_tau
    global Input_resistance
    global membrane_capacitance

    abf.setSweep(0, int(get_value("Voltage channel")))
    dt = abf.sweepX[1] - abf.sweepX[0]
    start1 = get_value("Passive stimuli start (s)")
    lenght1 = get_value("Passive stimuli lenght (s)")

    stim_start = int(float(start1) / dt)
    stim_end = int(stim_start + (float(lenght1) / dt))

    abf.setSweep(int(get_value("Sweep for passive parameters")), int(get_value("Voltage channel")))


    X1 = abf.sweepX[stim_start : int((stim_start + (0.05 / dt)))]           #calculate membrane tau
    Y1 = average_sweep[stim_start : int((stim_start + (0.05 / dt)))]

    p0 = (20, 10, 50)
    params, cv = scipy.optimize.curve_fit(monoExp, X1[::50], Y1[::50], p0, maxfev = 10000)
    m, t, b = params
    sampleRate = int(1 / dt / 1000)+1
    tauSec = ((1 / t) / sampleRate) * 1e6 / 20

    Rm = Input_resistance

    cap = tauSec / Rm *1000                                                 #calculate membrane capacitance

    print(tauSec)
    print(Rm)
    print(cap)
    membrane_tau = tauSec
    membrane_capacitance = cap
    """
        if tauSec > 100 or tauSec < 6:
            membrane_tau = 'nan'
        else:
            membrane_tau = tauSec

        if Rm > 10000 or Rm < 40 or membrane_tau == 'nan':
            Input_resistance = 'nan'
        else:
            Input_resistance = Rm
        if cap > 300 or cap < 5 or Input_resistance == 'nan':
            membrane_capacitance = 'nan'
        else:
            membrane_capacitance = cap
    except:
        pass
    """
    calc_resting()



# %%
def running_mean(x, N):                                                     #running mean to avoid measuring noise
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def calc_sag():
    global sag_amplitude
    global sag_timing


    abf.setSweep(0, int(get_value("Voltage channel")))
    dt = abf.sweepX[1] - abf.sweepX[0]
    start1 = get_value("Stimulus start (s)")
    lenght1 = get_value("Stimulus lenght (s)")

    stim_start = int(float(start1) / dt)
    stim_end = int(stim_start + (float(lenght1) / dt))

    abf.setSweep(int(get_value("Sweep for sag parameters")), int(get_value("Voltage channel")))

    moving_ave = running_mean(abf.sweepY, 1000)

    sag_min = np.min(moving_ave[stim_start : int((stim_start + (0.15 / dt)))]) - np.mean(abf.sweepY[int(stim_start + (float(lenght1)*0.8 / dt)) : stim_end])
    sag_timing1 = (np.argmin(moving_ave[stim_start : int((stim_start + (0.15 / dt)))])) * dt

    sag_amplitude = sag_min
    sag_timing = sag_timing1

# %%




# %%

def calc_freq():                                           #calculate mean and max firing frequency

    global mean_firing_frequency
    global max_firing_frequency

    abf.setSweep(0, int(get_value("Voltage channel")))
    dt = abf.sweepX[1] - abf.sweepX[0]
    start1 = get_value("Stimulus start (s)")
    lenght1 = get_value("Stimulus lenght (s)")

    stim_start = int(float(start1) / dt)
    stim_end = int(stim_start + (float(lenght1) / dt))

    abf.setSweep(int(get_value("Sweep for frequency calculations")), int(get_value("Voltage channel")))

    AP_count = 0
    AP_list = []

    for i in range(stim_start, stim_end):
        if abf.sweepY[i] > -20:
            if len(AP_list) > 0 and abs(AP_list[-1] - i) > 150: 
                AP_count += 1
                AP_list.append(i)
            if len(AP_list) == 0:
                AP_count += 1
                AP_list.append(i)
    
    mean_freq = AP_count / float(lenght1)

    max_freq = 1/((AP_list[1] - AP_list[0]) * dt)
    print(AP_list[1], AP_list[0], dt)

    mean_firing_frequency = mean_freq
    max_firing_frequency = max_freq



# %%
def calc_acc_ratio():

    global accommodation_ratio

    abf.setSweep(0, int(get_value("Voltage channel")))
    dt = abf.sweepX[1] - abf.sweepX[0]
    start1 = get_value("Stimulus start (s)")
    lenght1 = get_value("Stimulus lenght (s)")

    stim_start = int(float(start1) / dt)
    stim_end = int(stim_start + (float(lenght1) / dt))

    abf.setSweep(int(get_value("Sweep for accommodation ratio")), int(get_value("Voltage channel")))
                                               #calculate accommodation ratio

    AP_list = []

    for i in range(stim_start, stim_end):
        if abf.sweepY[i] > -20:
            if len(AP_list) > 0 and abs(AP_list[-1] - i) > 150: 
                AP_list.append(i)
            if len(AP_list) == 0:
                AP_list.append(i)

    accommodation_ratio1 = (AP_list[-1] - AP_list[-2]) / (AP_list[1] - AP_list[0])

    accommodation_ratio = accommodation_ratio1

# %%

def spike_scaling():

    global frequency_list
    global current_list

    abf.setSweep(0, int(get_value("Voltage channel")))
    dt = abf.sweepX[1] - abf.sweepX[0]

    start1 = get_value("Stimulus start (s)")
    lenght1 = get_value("Stimulus lenght (s)")

    stim_start = int(float(start1) / dt)
    stim_end = int(stim_start + (float(lenght1) / dt))

    sweep = 0
    freq_list = []
    curr_list =[]
    for i in range(abf.sweepCount):
        sweep = i
        abf.setSweep(sweep, int(get_value("Voltage channel")))

        AP_list = []

        for k in range(stim_start, stim_end):
            if abf.sweepY[k] > -20:
                if len(AP_list) > 0 and abs(AP_list[-1] - k) > 100: 
                    AP_list.append(k)
                if len(AP_list) == 0:
                    AP_list.append(k)

        if len(AP_list) == 0:
            freq_list.append(0)
        else:
            freq_list.append(len(AP_list) / float(lenght1))

        abf.setSweep(i, int(get_value("Command channel")))
        curr_list.append(np.mean(abf.sweepY[stim_start : (stim_start + 100)]))

    frequency_list = freq_list
    current_list = curr_list

    comp_list = np.transpose(np.vstack((np.asarray(current_list), np.asarray(frequency_list))))
    
    try:
        np.savetxt(get_value("Working directory") + "freqs.txt", comp_list)
    except:
        np.savetxt("freqs.txt", comp_list)

    with window("f/I", width=380, height=377):
        #add_button("Plot f/I", callback=plot_fI_callback)
        #add_same_line(spacing=13, name="sameline6")
        #add_same_line(spacing=10, name="sameline7")
        add_plot("Lineplot", height=-1)
        add_line_series("Lineplot", "frequency", list(current_list), list(frequency_list), weight=10, color=[232, 163, 33, 100])
        
        set_window_pos("f/I", 1040, 380)

    #return freq_list, curr_list

#%%



def calc_all_spike_params():

    peak = 'nan'
    hwdt = 'nan'
    AHP1 = 'nan'
    thresh = 'nan'
    dvdt_max1 = 'nan'

    try:
        f=open(get_value("Working directory") + 'all_spike_params.txt','a')
    except:
        f=open('all_spike_params.txt','a')

    f.write("sweep" + "\t" +
            "#AP" + "\t" +
            "AP peak" + "\t" + "\t" +
            "AP hwdt" + "\t" +
            "AHP" + "\t" +"\t" +
            "threshold" + "\t" +
            "dV/dt max" + '\n'
            )

    abf.setSweep(0, int(get_value("Voltage channel")))
    dt = abf.sweepX[1] - abf.sweepX[0]
    start1 = get_value("Stimulus start (s)")
    lenght1 = get_value("Stimulus lenght (s)")

    stim_start = int(float(start1) / dt)
    stim_end = int(stim_start + (float(lenght1) / dt))

    for sweep in range(abf.sweepCount):
        abf.setSweep(int(sweep), int(get_value("Voltage channel")))
        threshat = 0
        half2 = 0
        i = 0
        flag_counter = 0
        ap_counter = 0
        for i in range(int(float(lenght1) / dt)):
            if abf.sweepY[stim_start + i] > -20 and flag_counter > (0.002 / dt):
                st1 = int(stim_start + i)                               #calc peak
                st2 = int(stim_start + i + (0.002 / dt))
                peak = np.max(abf.sweepY[st1 : st2])
                peakat = np.argmax(abf.sweepY[st1 : st2]) + st1           

                dvdt = np.gradient(abf.sweepY, dt * 1000)                 #calculate dvdt_max
                st1 = int(stim_start + i - (0.002 / dt))
                st2 = int(stim_start + i + (0.002 / dt))
                dvdt_max1 = np.max(dvdt[st1 : st2])

                st1 = int(stim_start + i)                               #calculate threshold
                for k in range(int(0.002 / dt)):
                    if dvdt[st1 - k] < 50:
                        thresh = abf.sweepY[st1 - k]
                        threshat = st1 - k
                        break

                st1 = int(stim_start + i) 
                st2 = int(stim_start + i + (0.006 / dt))                #calc AHP
                AHP1 = abs(np.min(abf.sweepY[st1 : st2]) - thresh)
                
                half1 = (peakat - threshat) / 2 + threshat
                st1 = peakat
                st2 = peakat + int(0.002 / dt)
                for i in range(st1,st2):
                    if abf.sweepY[i] < abf.sweepY[int(half1)]:
                        half2 = i
                        break
                hwdt = (half2 - half1) * dt *1000 

                f.write(str(sweep + 1) + "\t" +
                        str(ap_counter) + "\t" +
                        str(peak) + "\t" +
                        str(hwdt) + "\t" +
                        str(AHP1) + "\t" +
                        str(thresh) + "\t" +
                        str(dvdt_max1) + '\n'
                        )
                

                ap_counter += 1
                flag_counter = 0
            flag_counter += 1    
                #break
    f.close()

        






# %%

#%%

def savetxt():
    try:
        f=open(get_value("Working directory") + 'results.txt','a')
    except:
        f=open('results.txt','a')

    f.write(str(AP_peak) + "\t" +
            str(AP_threshold) + "\t" +
            str(AP_hwdt) + "\t" +
            str(AHP) + "\t" +
            str(dvdt_max) + "\t" +
            str(membrane_tau) + "\t" +
            str(membrane_capacitance) + "\t" +
            str(Input_resistance) + "\t" +
            str(resting_membrane_potential) + "\t" +
            str(sag_amplitude) + "\t" +
            str(sag_timing) + "\t" +
            str(mean_firing_frequency) + "\t" +
            str(max_firing_frequency) + "\t" +
            str(rheobase) + "\t" +
            str(accommodation_ratio) + '\n'
            )
    f.close()

# %%

global_counter = 0

def print_res():
    global global_counter
    global_counter += 1
    window_name1 = "Analysis results" + str(global_counter)
    with window(window_name1, width=520, height=338):
        set_window_pos(window_name1, 520, 380)

        
        text1 = "AP peak: "+ "\t" + "\t" + "\t" + "\t" + "   " +str(AP_peak) + "\n"
        text2 = "AP threshold: "+ "\t" + "\t" + "\t" + "  "+ str(AP_threshold) + "\n"
        text3 = "AP hwdt: "+ "\t" + "\t" + "\t" + "\t" + "   " +str(AP_hwdt) + "\n"
        text4 = "AHP: "+ "\t" + "\t" + "\t" + "\t" + "\t"+ "  "+ " " + str(AHP) + "\n"
        text5 = "dV/dt maximum: "+ "\t" + "\t" + "\t" + " " +str(dvdt_max) + "\n"
        text6 = "membrane tau: "+ "\t" + "\t" + "\t"+ "  " + str(membrane_tau) + "\n"
        text7 = "membrane capacitance: "+ "\t" + "  " + str(membrane_capacitance) + "\n"
        text8 = "input resistance: "+ "\t" + "\t"+ "  " + str(Input_resistance) + "\n"
        text9 = "resting membrane potential: "+ str(resting_membrane_potential) + "\n"
        text10 = "sag amplitude: "+ "\t" + "\t" + "     " + str(sag_amplitude) + "\n"
        text11 = "sag timing: "+ "\t" + "\t" + "\t" + "\t" + str(sag_timing) + "\n"
        text12 = "mean firing frequency: "+ "     " + str(mean_firing_frequency) + "\n"
        text13 = "maximum firing frequency: "+ "  " + str(max_firing_frequency) + "\n"
        text14 = "rheobase: "+ "\t" + "\t" + "\t" + "\t" + "  "+ str(rheobase) + "\n"
        text15 = "accommodation ratio: "+ "\t" + "   " + str(accommodation_ratio) + "\n"

        add_text(text1)
        add_text(text2)
        add_text(text3)
        add_text(text4)
        add_text(text5)
        add_text(text6)
        add_text(text7)
        add_text(text8)
        add_text(text9)
        add_text(text10)
        add_text(text11)
        add_text(text12)
        add_text(text13)
        add_text(text14)
        add_text(text15)

# %%

def calc_all(sender, data):

    set1()
    print_res()

    try:
        calc_acc_ratio()
    except:
        pass
    try:
        calc_freq()
    except:
        pass
    try:
        calc_resting()
    except:
        pass
    try:
        calc_pas_params()
    except:
        pass
    
    try:
        calc_spike_params()
    except:
        pass
    try:
        calc_sag()
    except:
        pass
    savetxt()
    print_res()

def open_myabf(sender, data):
    global abf
    if get_value("Working directory")[0] == "e":
        abf_name = get_value("##file_name1")
    else:
        abf_name = str(get_value("Working directory")) + str(get_value("##file_name1"))

    abf = pyabf.ABF(str(abf_name))
    abf = pyabf.ABF(str("single.abf"))
    try:
        abf.setSweep(1,1)
    except:
        pass

#%%

frequency_list = []
current_list = []
#%%



set_main_window_size(1440, 760)
set_global_font_scale(1.25)
set_theme("Gold")
set_style_window_padding(30,30)



def save1_callback(sender, data):
    print("Save Clicked")


with window("Main Window", width=520, height=717):
    add_drawing("logo1", width=520, height=150) #create some space for the image
    draw_image("logo1", "OV_logo2.jpg", pmin=[0,0], pmax=[480, 150])
    set_window_pos("Main Window", 0, 0)
    #add_text("Hello world")
    #add_separator()
    #add_spacing(count=12)
    add_input_text("Working directory", default_value="eg. ..folder\ ", width=150)
    with tooltip("Working directory", "Tooltip ID_0"):
        add_text("make sure to include \ at the end")
    #add_input_text("##w_dir", default_value="eg. C:\Users\..\folder\ ", width=150)
    add_input_text("##file_name1", default_value="Open this file", width=150)
    with tooltip("##file_name1", "Tooltip ID_01"):
        add_text("type in the .abf extension as well")
    add_same_line(spacing=10, name="sameline1")
    add_button("Open", callback=open_myabf)
    add_same_line(spacing=80, name="sameline2")
    add_spacing(count=12)

    add_input_text("Stimulus start (s)", default_value="0.36256", width=50)
    add_input_text("Stimulus lenght (s)", default_value="0.3", width=50)
    add_input_text("Passive stimuli start (s)", default_value="0.06256", width=50)
    add_input_text("Passive stimuli lenght (s)", default_value="0.2", width=50)
    add_input_text("Voltage channel", default_value="0", width=50)
    add_input_text("Command channel", default_value="1", width=50)

    add_spacing(count=12)
    add_separator()
    add_spacing(count=12)

    add_input_text("Sweep for spike parameters", default_value="3", width=50)
    with tooltip("Sweep for spike parameters", "Tooltip ID_1"):
        add_text("Select the first subthreshold sweep")
    add_input_text("Sweep for passive parameters", default_value="1", width=50)
    with tooltip("Sweep for passive parameters", "Tooltip ID_2"):
        add_text("Select the sweep with the smallest stimuli")
    add_input_text("Sweep for sag parameters", default_value="1", width=50)
    with tooltip("Sweep for sag parameters", "Tooltip ID_3"):
        add_text("Select a hyperpolarized sweep")
    add_input_text("Sweep for frequency calculations", default_value="5", width=50)
    with tooltip("Sweep for frequency calculations", "Tooltip ID_4"):
        add_text("Select a subthreshold sweep with high stimuli")
    add_input_text("Sweep for accommodation ratio", default_value="5", width=50)
    with tooltip("Sweep for accommodation ratio", "Tooltip ID_5"):
        add_text("Preferrably the same sweep as for frequency calculations")
    add_spacing(count=12)
    add_button("Calculate all", callback=calc_all)
    add_button("Calculate f/I", callback=spike_scaling)
    add_button("Measure every spike", callback=calc_all_spike_params)

from math import cos, sin


def plot_callback(sender, data):

    for i in range(100):
        try:
            delete_series(plot="Recording", series="sweep" + str(i))
        except:
            pass
  

    xlist = [[] for i in range(abf.sweepCount)]
    ylist = [[] for i in range(abf.sweepCount)]
    for i in range(abf.sweepCount):
        abf.setSweep(i,int(get_value("Voltage channel")))
        xlist[i] = list(abf.sweepX)
        ylist[i] = list(abf.sweepY)

        plot_sweep_name = "sweep" + str(i)

        add_line_series("Recording", plot_sweep_name, xlist[i], ylist[i], weight=2, color=[232, 163, 33, 100])

        


def plot_sweep_callback(sender, data):

    for i in range(100):
        try:
            delete_series(plot="Recording", series="sweep" + str(i))
        except:
            pass

    sweep_name = get_value("##sweep_name")

    if sweep_name != "Sweep#":

        

        xlist = [[] for i in range(abf.sweepCount)]
        ylist = [[] for i in range(abf.sweepCount)]
        for i in range(abf.sweepCount):
            abf.setSweep(i,int(get_value("Voltage channel")))
            xlist[i] = list(abf.sweepX)
            ylist[i] = list(abf.sweepY)

            plot_sweep_name = "sweep" + str(i)
            if i == int(sweep_name):
                continue
            else:
                add_line_series("Recording", plot_sweep_name, xlist[i], ylist[i], weight=2, color=[120, 120, 120, 100])
        
        plot_sweep_name = "sweep" + str(int(sweep_name))
        abf.setSweep(int(sweep_name),1)
        add_line_series("Recording", plot_sweep_name, xlist[int(sweep_name)], ylist[int(sweep_name)], weight=2, color=[232, 163, 33, 100])




with window("Firing pattern", width=920, height=377):
    add_button("Plot firing pattern", callback=plot_callback)
    add_same_line(spacing=135, name="sameline4")
    add_input_text("##sweep_name", default_value="Sweep#", width=60)
    add_same_line(spacing=10, name="sameline5")
    add_button("Plot sweep", callback=plot_sweep_callback)

    add_plot("Recording", height=-1)
    set_window_pos("Firing pattern", 520, 0)


"""
with window("f/I", width=380, height=377):
    #add_button("Plot f/I", callback=plot_fI_callback)
    #add_same_line(spacing=13, name="sameline6")
    #add_same_line(spacing=10, name="sameline7")
    add_plot("Lineplot", height=-1)
    add_line_series("Lineplot", "frequency", list(frequency_list), list(current_list), weight=2, color=[232, 163, 33, 100])
    
    set_window_pos("f/I", 1040, 380)
"""



with window("Analysis results", width=520, height=338):
    set_window_pos("Analysis results", 520, 380)

    text1 = "AP peak: "+ "\t" + "\t" + "\t" + "\t" + "   " +str(AP_peak) + "\n"
    text2 = "AP threshold: "+ "\t" + "\t" + "\t" + "  "+ str(AP_threshold) + "\n"
    text3 = "AP hwdt: "+ "\t" + "\t" + "\t" + "\t" + "   " +str(AP_hwdt) + "\n"
    text4 = "AHP: "+ "\t" + "\t" + "\t" + "\t" + "\t"+ "  "+ " " + str(AHP) + "\n"
    text5 = "dV/dt maximum: "+ "\t" + "\t" + "\t" + " " +str(dvdt_max) + "\n"
    text6 = "membrane tau: "+ "\t" + "\t" + "\t"+ "  " + str(membrane_tau) + "\n"
    text7 = "membrane capacitance: "+ "\t" + "  " + str(membrane_capacitance) + "\n"
    text8 = "input resistance: "+ "\t" + "\t"+ "  " + str(Input_resistance) + "\n"
    text9 = "resting membrane potential: "+ str(resting_membrane_potential) + "\n"
    text10 = "sag amplitude: "+ "\t" + "\t" + "     " + str(sag_amplitude) + "\n"
    text11 = "sag timing: "+ "\t" + "\t" + "\t" + "\t" + str(sag_timing) + "\n"
    text12 = "mean firing frequency: "+ "     " + str(mean_firing_frequency) + "\n"
    text13 = "maximum firing frequency: "+ "  " + str(max_firing_frequency) + "\n"
    text14 = "rheobase: "+ "\t" + "\t" + "\t" + "\t" + "  "+ str(rheobase) + "\n"
    text15 = "accommodation ratio: "+ "\t" + "   " + str(accommodation_ratio) + "\n"

    add_text(text1)
    add_text(text2)
    add_text(text3)
    add_text(text4)
    add_text(text5)
    add_text(text6)
    add_text(text7)
    add_text(text8)
    add_text(text9)
    add_text(text10)
    add_text(text11)
    add_text(text12)
    add_text(text13)
    add_text(text14)
    add_text(text15)





#%%
start_dearpygui()

#%%

# %%

# %%
