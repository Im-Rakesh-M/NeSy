# test_day2.py
import asyncio
from agents.message_bus import MessageBus
from agents.logistics_agent import LogisticsAgent
from agents.maintenance_agent import MaintenanceAgent
from agents.inventory_agent import InventoryAgent
from agents.hr_agent import HRAgent
from agents.regulatory_agent import RegulatoryAgent
from agents.sop_agent import SOPAgent

async def main():
    bus = MessageBus()
    
    # Instantiate agents to register subscribers on the bus
    logistics = LogisticsAgent(bus)
    maintenance = MaintenanceAgent(bus)
    inventory = InventoryAgent(bus)
    hr = HRAgent(bus)
    reg = RegulatoryAgent(bus)
    sop = SOPAgent(bus)
    
    print("\n--- Simulating Scenario 1: Late Delivery + Low Buffer ---")
    mock_payload = {
        "order_id": "ORDER_TEST_99",
        "line_id": "LINE_A",
        "prob_late": 0.92,
        "prediction_set": [1] # Statistical delay guaranteed
    }
    
    await bus.publish("DELIVERY_RISK", mock_payload)
    await asyncio.sleep(2) # Give async loop processing time to execute

if __name__ == "__main__":
    asyncio.run(main())