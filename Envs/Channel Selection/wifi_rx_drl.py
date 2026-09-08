#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Wifi Rx
# GNU Radio version: 3.10.1.1

import random
import pmt
# from gnuradio import network
from gnuradio import zeromq

from packaging.version import Version as StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from PyQt5.QtCore import QObject, pyqtSlot
from gnuradio import qtgui
from gnuradio.filter import firdes
import sip
from gnuradio import blocks
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import gr
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import gr, blocks
from gnuradio import uhd
import time
from gnuradio.qtgui import Range, RangeWidget
from PyQt5 import QtCore
import ieee802_11
import numpy as np

from agent_rx import Agent
from gnuradio import qtgui
import h5py
from tqdm import tqdm


class feedback_generator(gr.basic_block):
    def __init__(self):
        gr.basic_block.__init__(self,
            name="feedback_generator",
            in_sig=[], out_sig=[])
        
        # Register an output message port
        self.message_port_register_out(pmt.intern("out"))

    def send_feedback(self, channel_flag:int):
        # mod_flag = random.randint(0, 4)  # Generate a random modulation flag
        pdu = pmt.cons(pmt.PMT_NIL, pmt.from_long(channel_flag))  # Create a PDU
        # Send the PDU to the "out" message port
        self.message_port_pub(pmt.intern("out"), pdu)
        print("Sent feedback channel flag:", channel_flag)

class csi_reader(gr.sync_block):
    def __init__(self, vector_size = 52, ):
        gr.sync_block.__init__(self,
                               name="csi_reader",
                               in_sig=[(np.float32, vector_size)],  # Input: float vector
                               out_sig=[])  # No output (purely for printing)
        self.incoming_csi = None
        self.message_port_register_out(pmt.intern("csi"))

    def work(self, input_items, output_items):
        if len(input_items[0]) > 0:
            self.incoming_csi = input_items
        else:
            self.incoming_csi = None
        # print("CSI Reader: Received CSI data:", self.incoming_csi)
        self.message_port_pub(pmt.intern("csi"), pmt.to_pmt(self.incoming_csi))
        return len(input_items[0])  # Return the number of processed vectors

class MacSeqReader(gr.basic_block):
    def __init__(self):
        gr.basic_block.__init__(self,
                                name="MacSeqReader",
                                in_sig=None,
                                out_sig=None)
        # Register an input message port called "in"
        self.message_port_register_in(pmt.intern("in"))
        # Set the message handler for the "in" port
        self.set_msg_handler(pmt.intern("in"), self.handle_msg)
        self.last_seq = None

    def handle_msg(self, msg):
        """
        This message handler extracts the sequence number from the metadata of the message.
        It expects the metadata (pmt.car(msg)) to contain a dictionary with a key "seq".
        """
        meta = pmt.car(msg)
        # **You can print the metadata dictionary to see its contents
        # Attempt to extract the sequence number from the metadata dictionary.
        seq_val = pmt.dict_ref(meta, pmt.intern("sequence number"), pmt.PMT_NIL)
        if pmt.is_number(seq_val):
            self.last_seq = pmt.to_long(seq_val)
            print("MacSeqReader: Received sequence number:", self.last_seq)
        else:
            print("MacSeqReader: Sequence number not found in metadata!")

    def get_last_seq(self):
        """Return the last sequence number that was received."""
        return self.last_seq


class RLTrigger(gr.basic_block):
    def __init__(self, parent, history_len = 10, threshold=100):
        gr.basic_block.__init__(self,
                                name="RLTrigger",
                                in_sig=None,
                                out_sig=None)
        self.parent = parent
        self.threshold = threshold
        # self.state_tpye = state_type
        self.history_len = history_len
        self.sample_count = 0
        self.csi_input = None
        self.csi_averager = np.zeros((52, history_len), dtype=np.float32)
        self.processing = False
        # self.csi_matrix = initial_state # shape: (52+1, channel_count)
        self.message_port_register_in(pmt.intern("out"))
        self.message_port_register_in(pmt.intern("csi"))
        self.set_msg_handler(pmt.intern("csi"), self.handle_csi_msg)

    def handle_csi_msg(self, msg):
        if self.processing:
            return
        # Get the CSI state (could be just the last vector or the full history)
        self.csi_input = pmt.to_python(msg)
        self.sample_count += 1
        # print("RLTrigger: Received CSI data:", self.csi_input)
        print("CUrrent sample count:", self.sample_count)
        if self.sample_count >= self.threshold-self.history_len:
            self.csi_averager[:,self.threshold-self.sample_count-1] = np.array(self.csi_input)
        if self.sample_count >= self.threshold:
            update_state = np.mean(self.csi_averager, axis=1) # size: 52
            self.sample_count = 0
            self.processing = True
            self.parent.process_rl(update_state)

class wifi_rx(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Wifi Rx")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Wifi Rx")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "wifi_rx")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.window_size = window_size = 48
        self.sync_length = sync_length = 320
        self.samp_rate = samp_rate = 10e6
        self.lo_offset = lo_offset = 0
        self.gain = gain = 0.85 # Higher means greater strength of power
        self.chan_est = chan_est = 0 # Channel estimation Default: LS
        self.csi_len = 52 # CSI vector length = 48 subcarrier + 4 pilots = 52
        self.csi_history_len = 8 #  Most recent 10 CSI to be used in state: to be averaged
        self.channel_mapping: dict[int,int] = {
            0: 5170000000, # original 5.17GHz
            1: 5200000000,
            2: 5270000000,
            3: 5320000000,
            4: 5720000000,
            5: 2412000000,
            6: 2442000000,
            7: 2472000000,
            # 0: 5620000000,
        } # 0-8: 5GHz channels 5.17G-5.72G --> 802.11a, interval = 500MHz
        self.channel_flag = 0 # Current channel choice
        self.num_channel = len(self.channel_mapping)
        self.freq = freq = self.channel_mapping[self.channel_flag] # Obtained from mapping dict by channle flag
        self.policy_freq = 80 # How many packets to trigger the RL agent
        ##################################################
        # Blocks
        ##################################################
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(('', "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
        )
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_center_freq(uhd.tune_request(freq, rf_freq = freq - lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)
        self.uhd_usrp_source_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_source_0.set_normalized_gain(gain, 0)
        self.print_initial_settings()
        # No synchronization enforced.

        # Receiving signal plot ----------------------------------------------
        self.qtgui_time_sink_x_0 = qtgui.time_sink_f(
            1024, #size
            samp_rate, #samp_rate
            "", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0.set_y_axis(-1, 1)

        self.qtgui_time_sink_x_0.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_0.enable_tags(True)
        self.qtgui_time_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_0.enable_autoscale(False)
        self.qtgui_time_sink_x_0.enable_grid(False)
        self.qtgui_time_sink_x_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0.enable_stem_plot(False)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_time_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_win = sip.wrapinstance(int(self.qtgui_time_sink_x_0.qwidget()), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_0_win)
        # Finish signal plot ----------------------------------------------

        # Constellation plot ----------------------------------------------
        self.qtgui_const_sink_x_0 = qtgui.const_sink_c(
            48*10, #size
            "", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_const_sink_x_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_0.set_x_axis(-2, 2)
        self.qtgui_const_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0.enable_grid(True)
        self.qtgui_const_sink_x_0.enable_axis_labels(True)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "red", "red", "red",
            "red", "red", "red", "red", "red"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_win = sip.wrapinstance(int(self.qtgui_const_sink_x_0.qwidget()), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_const_sink_x_0_win)
        # Finish Constellation plot ----------------------------------------------

        self.pdu_pdu_to_tagged_stream_0 = blocks.pdu_to_tagged_stream(blocks.complex_t, 'packet_len')
        self.ieee802_11_sync_short_0 = ieee802_11.sync_short(0.56, 2, False, False)
        self.ieee802_11_sync_long_0 = ieee802_11.sync_long(sync_length, False, False)
        self.ieee802_11_parse_mac_0 = ieee802_11.parse_mac(False, True)
        self.ieee802_11_extract_csi_0 = ieee802_11.extract_csi()
        self.ieee802_11_frame_equalizer_0 = ieee802_11.frame_equalizer(chan_est, freq, samp_rate, False, False)
        self.ieee802_11_decode_mac_0 = ieee802_11.decode_mac(True, False)
        self.fft_vxx_0 = fft.fft_vcc(64, True, window.rectangular(64), True, 1)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, 64)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_moving_average_xx_1 = blocks.moving_average_cc(window_size, 1, 4000, 1)
        self.blocks_moving_average_xx_0 = blocks.moving_average_ff(window_size  + 16, 1, 4000, 1)
        self.blocks_divide_xx_0 = blocks.divide_ff(1)
        self.blocks_delay_0_0 = blocks.delay(gr.sizeof_gr_complex*1, 16)
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex*1, sync_length)
        self.blocks_conjugate_cc_0 = blocks.conjugate_cc()
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(1)
        self.blocks_complex_to_mag_0 = blocks.complex_to_mag(1)
        
        # Newly added blocks
        print("Initializing Center Frequency:", self.freq)
        self.set_freq(self.freq) 
        print("Initializing MAC Seq Reader...")
        self.mac_seq_reader = MacSeqReader() # Read sequence number + meta data
        print("Initializing CSI Reader...")
        self.blocks_complex_to_mag_1 = blocks.complex_to_mag(self.csi_len) # Raw CSI is complex, need to convert to magnitude
        self.csi_reader = csi_reader(self.csi_len)
        print("Initializing RL Trigger...")
        self.rl_trigger = RLTrigger(self, self.csi_history_len, self.policy_freq)
        print("Initializing Feedback Generator...")
        self.feedback_generator = feedback_generator()
        print("Initializing ZeroMQ Pub Msg Sink...")
        self.zeromq_pub_msg_sink_0 = zeromq.pub_msg_sink('tcp://*:54321', 100, True)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.feedback_generator, 'out'), (self.zeromq_pub_msg_sink_0, 'in'))
        self.msg_connect((self.ieee802_11_decode_mac_0, 'out'), (self.ieee802_11_parse_mac_0, 'in'))
        self.msg_connect((self.ieee802_11_parse_mac_0, 'out'), (self.ieee802_11_extract_csi_0, 'pdu in'))
        self.msg_connect((self.ieee802_11_frame_equalizer_0, 'symbols'), (self.pdu_pdu_to_tagged_stream_0, 'pdus'))
        self.msg_connect((self.csi_reader, "csi"), (self.rl_trigger, "csi"))
        self.msg_connect((self.ieee802_11_parse_mac_0, 'out'), (self.mac_seq_reader, 'in'))
        self.connect((self.blocks_complex_to_mag_0, 0), (self.blocks_divide_xx_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_moving_average_xx_0, 0))
        self.connect((self.blocks_conjugate_cc_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.blocks_delay_0, 0), (self.ieee802_11_sync_long_0, 1))
        self.connect((self.blocks_delay_0_0, 0), (self.blocks_conjugate_cc_0, 0))
        self.connect((self.blocks_delay_0_0, 0), (self.ieee802_11_sync_short_0, 0))
        self.connect((self.blocks_divide_xx_0, 0), (self.ieee802_11_sync_short_0, 2))
        self.connect((self.blocks_divide_xx_0, 0), (self.qtgui_time_sink_x_0, 0))
        self.connect((self.blocks_moving_average_xx_0, 0), (self.blocks_divide_xx_0, 1))
        self.connect((self.blocks_moving_average_xx_1, 0), (self.blocks_complex_to_mag_0, 0))
        self.connect((self.blocks_moving_average_xx_1, 0), (self.ieee802_11_sync_short_0, 1))
        self.connect((self.blocks_multiply_xx_0, 0), (self.blocks_moving_average_xx_1, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.ieee802_11_frame_equalizer_0, 0))
        self.connect((self.ieee802_11_frame_equalizer_0, 0), (self.ieee802_11_decode_mac_0, 0))
        self.connect((self.ieee802_11_sync_long_0, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.ieee802_11_sync_short_0, 0), (self.blocks_delay_0, 0))
        self.connect((self.ieee802_11_sync_short_0, 0), (self.ieee802_11_sync_long_0, 0))
        self.connect((self.pdu_pdu_to_tagged_stream_0, 0), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_complex_to_mag_squared_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_delay_0_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.ieee802_11_extract_csi_0, 0), (self.blocks_complex_to_mag_1, 0))
        self.connect((self.blocks_complex_to_mag_1, 0), (self.csi_reader, 0))
        
        
        # IF trigger by a static timer, the DRL agent will be called every 1 sec
        # self.feedback_timer = QtCore.QTimer(self)
        # self.feedback_timer.setInterval(1000)  # 1000 ms = 1 sec
        # self.feedback_timer.timeout.connect(self.send_feedback_call)
        # self.feedback_timer.start()

        # IF trigger by a counter, the DRL agent will be called every 100 packet, determined by self.policy_freq
        # See this part in: self.msg_connect((self.csi_reader, "csi"), (self.rl_trigger, "csi"))

        ##################################################
        # RL Agent
        ##################################################
        print("Initializing DRL agent...")
        self.state_matrix = np.ones((self.csi_len+1, self.num_channel), dtype=np.float32)
        self.agent = Agent(
            state_size = self.state_matrix.shape,  
            action_size = self.num_channel,
            ) # By default, using the vector size of CSI
        self.initializing = True
        self.init_index = 0

        self.rounds = 0
        self.terminate_round = 1000 # Total received packet count = self.policy_freq * self.terminate_round
        
        self.previous_state = np.copy(self.state_matrix)
        self.previous_action = 0
        self.reward = 0
        self.reward_list = []
        self.previous_seq = self.total_packet_count = 0 # for reward calculation
        print("State size:", self.state_matrix.shape)
        print("Action size:", self.num_channel)
        print("Policy frequency:", self.policy_freq)
        print("Termination rounds:", self.terminate_round)
        # print("Initialization complete.")


    # def send_feedback_call(self):
    #     print("current channel", self.channel_flag)
        # self.channel_flag = self.feedback_generator.send_feedback()
        # Update channle_flag, do not need to update the frequency here for sync issue
        self.offline_dataset_init()

    def process_rl(self, new_state):
        if self.initializing:
            # --- Initialization Phase: sequentially measure each channel ---
            print("Initialization: Scanning channel", self.init_index)
            # Update state matrix for the current channel
            self.state_matrix = self.state_matrix_update(self.state_matrix, self.init_index, new_state)
            self.init_index += 1
            print(len(self.channel_mapping))
            if self.init_index < len(self.channel_mapping):
                # Switch to next channel in the mapping
                new_freq = self.channel_mapping[self.init_index]
                self.set_freq(new_freq)
                print("Switching to channel index", self.init_index, "with frequency", new_freq)
                self.feedback_generator.send_feedback(self.init_index)
            else:
                print("Initialization complete.")
                self.initializing = False
                # Save initial state and action if needed:
                self.previous_state = np.copy(self.state_matrix)
                self.previous_action = self.channel_flag  # or set a default
                self.previous_seq = current_seq = self.mac_seq_reader.get_last_seq()
                self.seq_for_initial = current_seq
        else:
            # Obtain the current state matrix
            self.state_matrix = self.state_matrix_update(self.state_matrix, self.channel_flag, new_state)
            current_state = np.copy(self.state_matrix)

            # Take action, inform TX for channel change, and update the RX frequency
            #self.channel_flag = int(self.agent.select_action(current_state))
            self.channel_flag = int(self.channel_flag+1)%self.num_channel
            current_action = self.channel_flag
            #print("========================================================DRL agent selected channel flag:", current_action)
            #print("========================================================Current state matrix:", current_state, current_state.shape)
            self.feedback_generator.send_feedback(current_action)
            new_freq = self.channel_mapping.get(current_action, self.freq)  # Fallback to current freq if needed
            if new_freq != self.freq:
                self.set_freq(new_freq)
            else:
                pass # No need to change the frequency if same

            # Rounds checking
            self.rounds += 1
            done = 0
            if self.rounds >= self.terminate_round: done = 1

            # Count the reward: packet loss rate at current round (Optional: self.policy_freq*self.rounds/(current_seq-self.seq_for_initial) = total reward)
            current_seq = self.mac_seq_reader.get_last_seq()
            self.total_packet_count = current_seq - self.previous_seq
            self.reward = self.policy_freq/self.total_packet_count
            #print("====================================================================================Reward for current round:", self.reward,self.total_packet_count)
            #self.reward_list.append(self.reward) # For debugging 
            #self.agent.step(self.previous_state, self.previous_action, self.reward, current_state, done)
            self.save_offline_data(self.previous_state, self.previous_action, self.reward, current_state, done)

            # Update the previous state, action, and reward aux for next step
            self.previous_state = current_state
            self.previous_action = current_action
            self.previous_seq = current_seq
        self.rl_trigger.processing = False

    def save_offline_data(self, state, action, reward, next_action, done):
        file_path = "./wireless_roundrobin_heuristic.h5"
        with h5py.File(file_path, 'a') as f:
            states = f['states']
            actions = f['actions']
            rewards = f['rewards']
            next_states = f['next_states']
            dones = f['dones']
            if self.offline_pointer < len(states):
                states[self.offline_pointer] = state
                actions[self.offline_pointer] = action
                rewards[self.offline_pointer] = reward
                next_states[self.offline_pointer] = next_action
                dones[self.offline_pointer] = done
                print("############################################################Offline dataset updated at index", self.offline_pointer)
            else:
                print("############################################################Offline dataset is full.")
                print("Offline dataset is full.")
                print("############################################################Offline dataset is full.")
                print("############################################################Offline dataset is full.")
                print("############################################################Offline dataset is full.")
            self.offline_pointer += 1
        f.close()

    def offline_dataset_init(self):
        file_path = "./wireless_roundrobin_heuristic.h5"
        sample_data_size = 200000
        state_shape = (53, 8)
        action_dim = 8

        with h5py.File(file_path, 'w') as f:
            states = f.create_dataset('states', (sample_data_size,) + state_shape, dtype='float32')
            actions = f.create_dataset('actions', (sample_data_size, action_dim), dtype='int32')
            rewards = f.create_dataset('rewards', (sample_data_size,), dtype='float32')
            next_states = f.create_dataset('next_states', (sample_data_size,) + state_shape, dtype='float32')
            dones = f.create_dataset('dones', (sample_data_size,), dtype='bool')
        f.close()
        self.offline_pointer = 153502
    
    def state_matrix_update(self, state_matrix, channel_flag: int, new_state):
        state_matrix[1:, channel_flag] = new_state
        for i in range(self.num_channel):
            if i == channel_flag:
                state_matrix[0, i] = np.exp(0)
            else:
                state_matrix[0, i] *= np.exp(-1)
        return state_matrix
    

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "wifi_rx")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_window_size(self):
        return self.window_size

    def set_window_size(self, window_size):
        self.window_size = window_size
        self.blocks_moving_average_xx_0.set_length_and_scale(self.window_size  + 16, 1)
        self.blocks_moving_average_xx_1.set_length_and_scale(self.window_size, 1)

    def get_sync_length(self):
        return self.sync_length

    def set_sync_length(self, sync_length):
        self.sync_length = sync_length
        self.blocks_delay_0.set_dly(self.sync_length)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self._samp_rate_callback(self.samp_rate)
        self.ieee802_11_frame_equalizer_0.set_bandwidth(self.samp_rate)
        self.qtgui_time_sink_x_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_lo_offset(self):
        return self.lo_offset

    def set_lo_offset(self, lo_offset):
        self.lo_offset = lo_offset
        self._lo_offset_callback(self.lo_offset)
        self.uhd_usrp_source_0.set_center_freq(uhd.tune_request(self.freq, rf_freq = self.freq - self.lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.uhd_usrp_source_0.set_normalized_gain(self.gain, 0)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        # self._freq_callback(self.freq)
        self.ieee802_11_frame_equalizer_0.set_frequency(self.freq)
        self.uhd_usrp_source_0.set_center_freq(uhd.tune_request(self.freq, rf_freq = self.freq - self.lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)

    def get_chan_est(self):
        return self.chan_est

    def set_chan_est(self, chan_est):
        self.chan_est = chan_est
        self._chan_est_callback(self.chan_est)
        self.ieee802_11_frame_equalizer_0.set_algorithm(ieee802_11.Equalizer(self.chan_est))
    
    def print_initial_settings(self):
        print("USRP Initial Settings:")
        print("  Sample Rate       :", self.samp_rate)
        print("  Center Frequency  :", self.freq)
        print("  LO Offset         :", self.lo_offset)
        print("  Normalized Gain   :", self.gain)
        print("  Channel Mapping   :", self.channel_mapping)


def main(top_block_cls=wifi_rx, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
