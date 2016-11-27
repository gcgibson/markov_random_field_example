import numpy as np
import sys
import math
import time


class edgeStruct():
	def __init__(self,adj,nStates):
		edgeEnds = np.zeros((1,2))
		for i in range(len(edgeEnds)):
			edgeEnds[i][0] = i
			edgeEnds[i][1] = i+1
		V = range(3)
		E = []
		for i in range(27):
			E.append(i)
			E.append(i)
		self.edgeEnds = edgeEnds
		self.nNodes = 2
		self.nEdges = 1
		self.nStates =  (np.ones(2)*2)




def UGM_Decode_Exact(nodePot, edgePot, edgeStruct):
	nNodes,maxState = nodeMap.shape
	nEdges = edgeStruct.nEdges
	edgeEnds = edgeStruct.edgeEnds
	nStates = edgeStruct.nStates
	y = np.ones((nNodes,1))
	nodeLabels = np.ones((nNodes,1))
	maxPot = -1
	while 1:
		pot = UGM_ConfigurationPotential(y,nodePot,edgePot,edgeEnds)
		if pot > maxPot:
			maxPot = pot
			nodeLabels[0] = y[0]
			nodeLabels[1] = y[1]
		for yInd in range(nNodes):
			y[yInd] = y[yInd] + 1;
			if y[yInd] <= nStates[yInd]:
				break
			else:
				y[yInd] = 1
	

		if  yInd == nNodes -1 and y[-1] == 1:
			
			break

	return nodeLabels


def UGM_ConfigurationPotential(y,nodePot,edgePot,edgeEnds):
	nNodes = nodePot.shape[0]
	nEdges = edgeEnds.shape[0]
	pot = 1
	for n in range(nNodes):
		pot = pot*nodePot[n,int(y[n]-1)]
	for e in range(nEdges):
		n1 = edgeEnds[e,0]
		n2 = edgeEnds[e,1]
		pot = pot*edgePot[int(y[int(n1)]-1),int(y[int(n2)]-1),int(e)];
	return pot

def Infer_Exact(nodePot, edgePot, edgeStruct):
	nNodes,maxState = nodeMap.shape
	nEdges = edgeStruct.nEdges
	edgeEnds = edgeStruct.edgeEnds
	nStates = edgeStruct.nStates
	nodeBel = np.zeros((nodePot.shape))
	edgeBel = np.zeros((edgePot.shape))
	y = np.ones((nNodes))
	Z = 0
	i = 1
	while 1:

		pot = UGM_ConfigurationPotential(y,nodePot,edgePot,edgeEnds)

		for n in range(nNodes):
			nodeBel[n,int(y[n]-1)] += pot
		for e in range(nEdges):
			n1 = edgeEnds[e,0]
			n2 = edgeEnds[e,1]
			edgeBel[int(y[int(n1)]-1),int(y[int(n2)]-1),e] += pot
		Z += pot
		for yInd in range(nNodes):
			y[yInd] = y[yInd] + 1

			if y[yInd] <= nStates[yInd]:
				break;
			else:
				y[yInd] = 1
			
		
		
		if  yInd == nNodes -1 and y[-1] == 1:
			break

	nodeBel = np.array(nodeBel)/Z
	edgeBel = np.array(edgeBel)/Z
	logZ = math.log(Z)
	return nodeBel, edgeBel, logZ
def UGM_MRF_makePotentials(w,nodeMap,edgeMap,edgeStruct):
	nNodes,maxState = nodeMap.shape
	nEdges = edgeStruct.nEdges
	edgeEnds = edgeStruct.edgeEnds
	nStates = edgeStruct.nStates
	nodePot = np.zeros((nNodes,maxState))
	for n in range(nNodes):
		for s in range(len(nStates)):
			if nodeMap[n,s] == 0:
				nodePot[n,s]=1
			else:
				 nodePot[n,s] = math.exp(w[int(nodeMap[n,s])])
	edgePot = np.zeros((maxState,maxState,nEdges))
	for e in range(nEdges):
		n1 = edgeEnds[e,0]
		n2 = edgeEnds[e,1]
		for s1 in range(len(nStates)):
			for s2 in range(len(nStates)):
				if edgeMap[s1,s2,e] == 0:
								edgePot[s1,s2,e] = 1
				else:
								edgePot[s1,s2,e] = math.exp(w[int(edgeMap[s1,s2,e])])
	return nodePot,edgePot


def UGM_MRF_computeSuffStat(Y,nodeMap,edgeMap,edgeStruct):
	suffStat = np.zeros(3)
	for i in range(len(Y)):
	   y = Y[i,:]
	   for n in range(edgeStruct.nNodes):
		  if nodeMap[n,y[n]-1] > 0:
			 suffStat[int(nodeMap[n,y[n]-1])] = suffStat[int(nodeMap[n,y[n]-1])] + 1

	   for e in range(edgeStruct.nEdges):
		  n1 = int(edgeStruct.edgeEnds[e,0])
		  n2 = int(edgeStruct.edgeEnds[e,1])
		  if edgeMap[y[n1]-1,y[n2]-1,e] > 0:
			 suffStat[int(edgeMap[y[n1]-1,y[n2]-1,e])] = suffStat[int(edgeMap[y[n1]-1,y[n2]-1,e])] + 1


	return suffStat






def UGM_MRF_NLL(param1,param2,*args):
	nodePot,edgePot = UGM_MRF_makePotentials([0,param1,param2],nodeMap,edgeMap,edgeStruct_)
	nodeBel, edgeBel, logZ= Infer_Exact(nodePot,edgePot,edgeStruct_)

	NLL = np.dot(np.transpose(-1*np.array([[0],[param1],[param2]])),suffStat)[0] + nInstances*logZ
	#NLL = NLL[0][0]
	g = -suffStat
	for n in range(nNodes):
			for s in range(2):
				if nodeMap[n,s] > 0:
					g[int(nodeMap[n,s])] = g[int(nodeMap[n,s])] + nInstances*nodeBel[n,s];
	for e in range(nEdges):
			n1 = edgeStruct_.edgeEnds[e,0];
			n2 = edgeStruct_.edgeEnds[e,1];
			for s1 in range(2):
				for s2 in range(2):
					if edgeMap[s1,s2,e] > 0:
						g[int(edgeMap[s1,s2,e])] = g[int(edgeMap[s1,s2,e])] + nInstances*edgeBel[s1,s2,e]

	return NLL
def run(y):
	nInstances,nNodes = y.shape
	nStates = np.argmax(y)
	adj = np.zeros((nNodes,nNodes))
	for i in range(nNodes-1):
		adj[i,i+1] = 1
	adj = np.add(adj,np.transpose(adj))
	edgeStruct_ =edgeStruct(adj,nStates)

	maxState = int(np.amax(edgeStruct_.nStates))

	nodeMap = np.zeros((nNodes,maxState))
	nodeMap[:,0] = 1
	nEdges= edgeStruct_.nEdges
	edgeMap = np.zeros((maxState,maxState,nEdges))
	edgeMap[0,0,:] = 2
	edgeMap[1,1,:] = 2
	nParams = 2

	param1 = 0
	param2 = 0


	suffStat = UGM_MRF_computeSuffStat(y,nodeMap,edgeMap,edgeStruct_)
	NLL = UGM_MRF_NLL(param1,param2,[nInstances,suffStat,nodeMap,edgeMap,edgeStruct_])

	from scipy.optimize import minimize
	x0 = NLL
	ret= minimize(lambda x : UGM_MRF_NLL(x[0],x[1]), [0,0])
	optimal_w = ret['x']	

	nodePot,edgePot = UGM_MRF_makePotentials([0,optimal_w[0],optimal_w[1]],nodeMap,edgeMap,edgeStruct_)

	print ("Most likely assignment : \n")
	final_assignment = UGM_Decode_Exact(nodePot, edgePot, edgeStruct_)
	print (final_assignment[0][0],final_assignment[1][0])

if __name__=="__main__":
	#DEFINE YOUR DATA HERE
	y = [[1,1],[2,2]]
	y= np.array(y)
