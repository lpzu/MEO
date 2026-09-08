# Channel Selection Testbed (Wi-Fi Scenario)

GNU Radio flowgraphs for CSI-based adaptive channel selection over USRP. The
transmitter sends packets at a fixed rate; the receiver decodes them, tracks the
per-channel CSI, and periodically selects the channel to use next. The reward is
derived from the packet loss rate measured over each round.

## Hardware

Two USRP N210s (one transmitter, one receiver). Other USRP models work but the
sample rate and antenna settings in the flowgraphs may need adjusting.

## Dependencies

The SDR stack is large, platform-dependent, and not installable from PyPI.
**Readers must install it themselves**, matching the versions to their own
system:

- **UHD** and the USRP firmware images (`sudo uhd_images_downloader`), verified
  with `uhd_find_devices`
- **GNU Radio** 3.8 or later
- **[gr-foo](https://github.com/bastibl/gr-foo)** and
  **[gr-ieee802-11](https://github.com/bastibl/gr-ieee802-11)**, both built from
  source. Check out the branch matching your GNU Radio version (`maint-3.8` or
  `maint-3.10`); a mismatched branch compiles but fails at runtime.
- Python: `torch`, `numpy`, `h5py`

## Files

```
wifi_tx_drl.py   # transmitter flowgraph
wifi_rx_drl.py   # receiver flowgraph, CSI tracking and channel switching
agent_rx.py      # RL agent (network, replay buffer, action selection)
```

## Core: `process_rl` in `wifi_rx_drl.py` (line 441)

**This is the point where the policy is plugged in.** `process_rl` is called
once per decision round. After the initial scan of all channels it updates the
state matrix, chooses the next channel, signals the transmitter, retunes the
receiver, computes the reward from the packet count, and stores the transition.

As shipped it runs a round-robin baseline:

```python
self.channel_flag = int(self.channel_flag + 1) % self.num_channel
```

Replace that line with your own policy, for example the DQN agent in
`agent_rx.py`:

```python
self.channel_flag = int(self.agent.select_action(current_state))
```

or any offline-trained model that maps `current_state` to a channel index.

## Running

Start the receiver first, then the transmitter.

```bash
python wifi_rx_drl.py    # on the RX machine
python wifi_tx_drl.py    # on the TX machine
```