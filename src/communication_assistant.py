def communication_assistant(masterQ, assistantQs, masterID, n_clients):
  myQ = assistantQs[masterID]
  n_processes_done = 0
  memory = {}
  while n_processes_done != n_clients:
    if not myQ.empty():
      msg = myQ.get()
      print(masterID, msg['type'])
      if msg['type'] == 'write':
        memory['data'] = msg['data']
        masterQ.put({'type': 'answer', 'status': 'success'})

      elif msg['type'] == 'read':
        if msg['from'] == masterID:
          assistantQs[msg['to']].put({'type': 'read', 'from': masterID})

        else:
          if 'data' not in memory.keys():
            assistantQs[msg['from']].put({'type': 'answer', 'status':'fail'})
          else:
            assistantQs[msg['from']].put({'type': 'answer', 'status': 'success', 'data': memory['data']})
      
      elif msg['type'] == 'answer':
        if msg['status'] == 'success':
          masterQ.put({'type': 'answer', 'status': 'success', 'data': msg['data']})
        else:
          masterQ.put({'type': 'answer', 'status': 'fail'})

      elif msg['type'] == 'done':
        n_processes_done += 1