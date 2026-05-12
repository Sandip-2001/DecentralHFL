
# canister_id = "u6s2n-gx777-77774-qaaba-cai" 


from ic.client import Client
from ic.identity import Identity
from ic.agent import Agent
from ic.candid import encode, Types

# 1. Connect to the local Internet Computer replica
client = Client(url="http://127.0.0.1:4943")
identity = Identity()
agent = Agent(identity=identity, client=client)

# REPLACE THIS with your actual fl_server_backend Canister ID
canister_id = "uxrrr-q7777-77774-qaaaq-cai" 

# 2. Create the dummy arrays AND dummy KL Trust Scores
array1 = [1.0, 2.0, 3.0, 4.0]
kl1 = 0.1

array2 = [2.0, 3.0, 4.0, 5.0]
kl2 = 0.2

array3 = [3.0, 4.0, 5.0, 6.0]
kl3 = 0.3

array4 = [4.0, 5.0, 6.0, 7.0]
kl4 = 0.4

# 3. The Upgrade: Translate BOTH arguments into Candid Binary
def format_candid_args(arr, kl_score):
    # Argument 1: The Array (Vec Float64)
    arg1 = {'type': Types.Vec(Types.Float64), 'value': arr}
    
    # Argument 2: The KL Score (Float64)
    arg2 = {'type': Types.Float64, 'value': kl_score}
    
    # Encode them together in the exact order the Motoko function expects
    return encode([arg1, arg2])

print("Uploading arrays AND trust scores to the Internet Computer...")

# 4. Send the binary payload to the Motoko 'update' functions
agent.update_raw(canister_id, "client1", format_candid_args(array1, kl1))
agent.update_raw(canister_id, "client2", format_candid_args(array2, kl2))
agent.update_raw(canister_id, "client3", format_candid_args(array3, kl3))
agent.update_raw(canister_id, "client4", format_candid_args(array4, kl4))

print("Upload complete! Fetching the PubAug weighted average...")

# 5. Call the 'query' function
result = agent.query_raw(canister_id, "testAverage", encode([]))

# Extracting the actual array
averaged_array = result[0]['value']

print("Weighted PubAug Result from Blockchain:", averaged_array)