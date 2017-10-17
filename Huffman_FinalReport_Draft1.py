# Will Huffman
# BME 503
# Final Report

from brian2 import *
import matplotlib.pyplot as plt
import math
map_size = 100
global foodx, foody, predx, predy, food_count, bug_plot, food_plot, pred_plot, sr_plot, sl_plot
foodx = 50
foody = 50
predx = -50
predy = -50
food_count = 0
pred_count = 0


# Sensor neurons
a = 0.025
#a = 0.1
b = 0.2
c = -65
d = 0.5

I0ex = 750
#I0ex = 0
I0in = 750
#I0in = 0
tau_ampa=1.0*ms
tau_ampa_in = 20*ms
g_synpk_ex=0.2
g_synmaxval_ex=(g_synpk_ex/(tau_ampa/ms*exp(-1)))
g_synpk_in=0.02
g_synmaxval_in=(g_synpk_in/(tau_ampa_in/ms*exp(-1)))
Eex = 0 # excititory reversal potential
Ein = -100 # inhibitory reversal potential

sensor_eqs = '''
dv/dt = 0.04*v**2/ms + 5*v/ms + 140/ms - u/ms + I/ms : 1
du/dt = (a*(b*v - u))/ms : 1
I = magfood*I0ex / sqrt(((x-foodx)**2+(y-foody)**2)) + magpred*I0in / sqrt(((x-predx)**2+(y-predy)**2)) + Iex + Iin: 1

Iex = gex * (Eex - v) : 1
dgex/dt = -gex/tau_ampa + zex/ms : 1
dzex/dt = -zex/tau_ampa : 1

Iin = gin * (Ein - v) : 1
dgin/dt = -gin/tau_ampa_in + zin/ms : 1
dzin/dt = -zin/tau_ampa_in : 1

x : 1
y : 1
x_disp : 1
y_disp : 1
foodx : 1
foody : 1
predx : 1
predy : 1
magfood : 1
magpred : 1
'''

sensor_reset = '''
v = c
u = u + d
'''

sre = NeuronGroup(1, sensor_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = sensor_reset)
sre.v = c
sre.u = c*b
sre.x_disp = 5
sre.y_disp = 5
sre.x = sre.x_disp
sre.y = sre.y_disp
sre.foodx = foodx
sre.foody = foody
sre.predx = predx
sre.predy = predy
sre.magfood=1
sre.magpred = 0

sri = NeuronGroup(1, sensor_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = sensor_reset)
sri.v = c
sri.u = c*b
sri.x_disp = 5
sri.y_disp = 5
sri.x = sri.x_disp
sri.y = sri.y_disp
sri.foodx = foodx
sri.foody = foody
sri.predx = predx
sri.predy = predy
sri.magfood = 0
sri.magpred = 1

sle = NeuronGroup(1, sensor_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = sensor_reset)
sle.v = c
sle.u = c*b
sle.x_disp = -5
sle.y_disp = 5
sle.x = sle.x_disp
sle.y = sle.y_disp
sle.foodx = foodx
sle.foody = foody
sle.predx = predx
sle.predy = predy
sle.magfood=1
sle.magpred=0

sli = NeuronGroup(1, sensor_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = sensor_reset)
sli.v = c
sli.u = c*b
sli.x_disp = -5
sli.y_disp = 5
sli.x = sle.x_disp
sli.y = sle.y_disp
sli.foodx = foodx
sli.foody = foody
sli.predx = predx
sli.predy = predy
sli.magfood=0
sli.magpred=1

sbr = NeuronGroup(1, sensor_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = sensor_reset) #motor neuron
sbr.v = c
sbr.u = c*b
sbr.foodx = foodx
sbr.foody = foody
sbr.predx = predx
sbr.predy = predy
sbr.magfood=0
sbr.magfood=0

sbl = NeuronGroup(1, sensor_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = sensor_reset) #motor neuron
sbl.v = c
sbl.u = c*b
sbl.foodx = foodx
sbl.foody = foody
sbl.predx = predx
sbl.predy = predy
sbl.magfood=0
sbl.magpred=0

# The virtual bug - may need to adjust these

taum = 4*ms
base_speed = 12
turn_rate = 5*Hz

bug_eqs = '''
dmotorl/dt = -motorl/taum : 1
dmotorr/dt = -motorr/taum : 1

speed = (motorl + motorr)/2 + base_speed : 1
dangle/dt = (motorr - motorl)*turn_rate : 1

dx/dt = speed*cos(angle)*15*Hz : 1
dy/dt = speed*sin(angle)*15*Hz : 1
'''

bug = NeuronGroup(1, bug_eqs, clock=Clock(0.2*ms))
bug.angle = pi/2
bug.x = 0
bug.y = 0

# Synapses (sensors communicate with bug motor)
w = 10
syn_rr=Synapses(sre, sbl, clock=Clock(0.2*ms), model='''
                g_synmax:1
                ''',
		on_pre='''
		zex += g_synmax
		''')

syn_rr.connect(i=[0],j=[0])
syn_rr.g_synmax=g_synmaxval_ex

syn_rri=Synapses(sri, sbr, clock=Clock(0.2*ms), model='''
                g_synmax:1
                ''',
		on_pre='''
		zex += g_synmax
		''')

syn_rri.connect(i=[0],j=[0])
syn_rri.g_synmax=g_synmaxval_ex

syn_ll=Synapses(sle, sbr, clock=Clock(0.2*ms), model='''
                g_synmax:1
                ''',
		on_pre='''
		zex += g_synmax
		''')
		
syn_ll.connect(i=[0],j=[0])
syn_ll.g_synmax=g_synmaxval_ex

syn_lli=Synapses(sli, sbl, clock=Clock(0.2*ms), model='''
                g_synmax:1
                ''',
		on_pre='''
		zex += g_synmax
		''')
		
syn_lli.connect(i=[0],j=[0])
syn_lli.g_synmax=g_synmaxval_ex

syn_ll_inh = Synapses(sli, sle, clock=Clock(0.2*ms), model='''
                g_synmax:1
                ''',
		on_pre='''
		zin += g_synmax
		''')
		
syn_ll_inh.connect(i=[0],j=[0])
syn_ll_inh.g_synmax=g_synmaxval_in

syn_rr_inh = Synapses(sri, sre, clock=Clock(0.2*ms), model='''
                g_synmax:1
                ''',
		on_pre='''
		zin += g_synmax
		''')
		
syn_rr_inh.connect(i=[0],j=[0])
syn_rr_inh.g_synmax=g_synmaxval_in

syn_r = Synapses(sbr, bug, clock=Clock(0.2*ms), on_pre='motorr += w')
syn_r.connect(i=[0],j=[0])
syn_l = Synapses(sbl, bug, clock=Clock(0.2*ms), on_pre='motorl += w')
syn_l.connect(i=[0],j=[0])

f = figure(1)
bug_plot = plot(bug.x, bug.y, 'ko')
food_plot = plot(foodx, foody, 'b*')
pred_plot = plot(predx, predy, 'r*')
sr_plot = plot([0], [0], 'w')   # Just leaving it blank for now
sl_plot = plot([0], [0], 'w')
# Additional update rules (not covered/possible in above eqns)

pred_speed = 0.02

@network_operation()
def update_positions():
    global foodx, foody, predx, predy, food_count, pred_count, food_speed
    sre.x = bug.x + sre.x_disp*sin(bug.angle)+ sre.y_disp*cos(bug.angle) 
    sre.y = bug.y + - sre.x_disp*cos(bug.angle) + sre.y_disp*sin(bug.angle) 

    sle.x = bug.x +  sle.x_disp*sin(bug.angle)+sle.y_disp*cos(bug.angle)
    sle.y = bug.y  - sle.x_disp*cos(bug.angle)+sle.y_disp*sin(bug.angle) 
    
#    sr.x = bug.x + sr.x_disp*cos(bug.angle-pi/2) - sr.y_disp*sin(bug.angle-pi/2)
#    sr.y = bug.y + sr.x_disp*sin(bug.angle-pi/2) + sr.y_disp*cos(bug.angle-pi/2)
#
#    sl.x = bug.x + sl.*cos(bug.angle-pi/2) - sl.y_disp*sin(bug.angle-pi/2)
#    sl.y = bug.y + sl.x_disp*sin(bug.angle-pi/2) + sl.y_disp*cos(bug.angle-pi/2)



    if ((bug.x-foodx)**2+(bug.y-foody)**2) < 16:
	food_count += 1
	foodx = randint(-map_size+10, map_size-10)
	foody = randint(-map_size+10, map_size-10)
    
    if ((bug.x-predx)**2+(bug.y-predy)**2) < 16:
	pred_count += 1
	predx = randint(-map_size+10, map_size-10)
	predy = randint(-map_size+10, map_size-10)
    else:
        pred_bug_x = bug.x - predx
        pred_bug_y = bug.y - predy
        
        predx = predx + pred_speed * (pred_bug_x/sqrt(pred_bug_x**2+pred_bug_y**2))
        predy = predy + pred_speed * (pred_bug_y/sqrt(pred_bug_x**2+pred_bug_y**2))

    if (bug.x < -map_size):
        bug.x = -map_size
        bug.angle = pi - bug.angle
    if (bug.x > map_size):
	bug.x = map_size
	bug.angle = pi - bug.angle
    if (bug.y < -map_size):
	bug.y = -map_size
	bug.angle = -bug.angle
    if (bug.y > map_size):
	bug.y = map_size
	bug.angle = -bug.angle

    sre.foodx = foodx
    sre.foody = foody
    sri.predx = predx
    sri.predy = predy
    
    sle.foodx = foodx
    sle.foody = foody
    sli.predx = predx
    sli.predy = predy
    

@network_operation(dt=15*ms)
def update_plot():
    global foodx, foody, predx, predy, bug_plot, food_plot, pred_plot, sr_plot, sl_plot
    bug_plot[0].remove()
    food_plot[0].remove()
    pred_plot[0].remove()
    sr_plot[0].remove()
    sl_plot[0].remove()
    bug_x_coords = [bug.x, bug.x-2*cos(bug.angle), bug.x-4*cos(bug.angle)]    # ant-like body
    bug_y_coords = [bug.y, bug.y-2*sin(bug.angle), bug.y-4*sin(bug.angle)]
    bug_plot = plot(bug_x_coords, bug_y_coords, 'ko')     # Plot the bug's current position
    sr_plot = plot([bug.x, sre.x], [bug.y, sre.y], 'b')     #plot the antenna
    sl_plot = plot([bug.x, sle.x], [bug.y, sle.y], 'r')     #plot the antenna
    food_plot = plot(foodx, foody, 'b*')
    pred_plot = plot(predx, predy, 'r*')
    axis([-100,100,-100,100])
    draw()
    #print "."
    pause(0.01)

ML = StateMonitor(sle, ('v', 'I'), record=True)
MR = StateMonitor(sre, ('v', 'I'), record=True)
IL = StateMonitor(sli, ('v', 'I'), record=True)
IR = StateMonitor(sri, ('v', 'I'), record=True)
MBL = StateMonitor(sbl, ('v', 'I'), record=True)
MBR = StateMonitor(sbr, ('v', 'I'), record=True)
MB = StateMonitor(bug, ('motorl', 'motorr', 'speed', 'angle', 'x', 'y'), record = True)

ML_rate = PopulationRateMonitor(sle)
MR_rate = PopulationRateMonitor(sre)

run(10000*ms,report='text')

#figure(2)
#plot(ML.t/ms, ML.v[0],'r')
#plot(MR.t/ms, MR.v[0],'b')
#title('v')
#figure(3)
#plot(ML.t/ms, ML.I[0],'r')
#plot(MR.t/ms, MR.I[0],'b')
#title('I')
#figure(4)
#plot(MB.t/ms, MB.motorl[0],'r')
#plot(MB.t/ms, MB.motorr[0],'b')
#title('Motor')


plt.clf()
plt.plot(MB.x[0], MB.y[0])
plt.plot(foodx, foody, 'b*')
plt.plot(predx, predy, 'r*')
axis([-100,100,-100,100])
title('Path')
show()