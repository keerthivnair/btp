import numpy as np
from telemetry.communication import CommunicationTracker

def test_communication_tracking():
    tracker = CommunicationTracker()
    
    params = [np.zeros((10, 10), dtype=np.float32)] # 10 * 10 * 4 bytes = 400 bytes
    
    downlink = tracker.record_downlink(1, "client_0", params)
    assert downlink == 400
    
    uplink = tracker.record_uplink(1, "client_0", params)
    assert uplink == 400
    
    totals = tracker.get_client_totals("client_0")
    assert totals["uplink_bytes"] == 400
    assert totals["downlink_bytes"] == 400
    assert totals["total_bytes"] == 800
    
    round_totals = tracker.get_round_totals(1)
    assert round_totals["uplink_bytes"] == 400
    assert round_totals["downlink_bytes"] == 400
    assert round_totals["total_bytes"] == 800
