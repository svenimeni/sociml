# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:43:22 2018

@author: Sonic
"""
import pandas as pd
import xml.etree.cElementTree as et

import liga_data
import tensorflow as tf

def main(argv):
   
    print("***************************************")
    print("******            New Run         *****")
    print('\nDebug Loading')
    parsedXML = et.parse( "sampleData/Ergbeniss_Saison_2016.xml" )  
    
    for node in parsedXML.getroot():
       #  for child in node:
       #     print('Node found: ', child.tag)
        matchID = node.find('MatchID').text
        print('MacthID: ', matchID)
        team1 = node.find('Team1').find('TeamName').text
        team2 = node.find('Team2')
        for child in team2:
           print('Child of Team 2  found: ', child.tag)
        team2 = node.find('Team2').find('TeamName').text
        print('Match found: ', team1, team2)
        matchpairing = team1 + team2
        print('Match found: ', matchpairing)
        
        
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)