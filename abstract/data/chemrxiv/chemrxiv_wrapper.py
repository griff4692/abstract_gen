#!/usr/bin/env python3
'''
Created on Nov 2 16:12:00 2021

@author: bingyinh
'''

import requests
import dateutil.parser

def chemrxiv_api(**kwargs):
    '''
    A python wrapper for making API call to the ChemRxiv server.
    **kwargs takes the following optional keyword arguments
    
    :param term: term for search and filtering, API default value ""
    :type term: str
    
    :param limit: the number limit for returned items, no bigger than 50,
        API default value 10
    :type limit: int
    
    :param skip: the first N collection items to be excluded from a response body,
        API default value 0
    :type skip: int
    
    :param sort: the order by which the response sorts the returned items,
        valid values: "VIEWS_COUNT_ASC", "VIEWS_COUNT_DESC", "READ_COUNT_ASC",
                      "READ_COUNT_DESC", "RELEVANT_ASC", "RELEVANT_DESC",
                      "PUBLISHED_DATE_ASC", "PUBLISHED_DATE_DESC"
        API default value "PUBLISHED_DATE_DESC"
    :type sort: str
    
    :param author: query by the specified author
    :type author: str
    
    :param searchDateFrom: filter out the result published earlier than the
        specified date, must be in ISO 8601 format 'YYYY-mm-ddTHH:MM:SS.fffZ'
    :type searchDateFrom: str
    
    :param searchDateTo: filter out the result published later than the specified
        date, must be in ISO 8601 format 'YYYY-mm-ddTHH:MM:SS.fffZ'
    :type searchDateTo: str
    
    :param categoryIds: query by the category id, category information can be
        retrieved by /public-api/v1/categories
    :type categoryIds: List[str]
    
    :param subjectIds: query by the subject id, category information can be
        retrieved by /public-api/v1/subjects
    :type subjectIds: List[str]
    
    '''
    url = 'https://chemrxiv.org/engage/chemrxiv/public-api/v1/items'
    # configure keywords
    payload = {}
    query_params = {'term', 'limit', 'skip', 'sort', 'author', 'searchDateFrom',
                    'searchDateTo', 'categoryIds', 'subjectIds'}
    # if 'limit' is specified, make sure it's no bigger than 50
    if 'limit' in kwargs:
        kwargs['limit'] = min(kwargs['limit'],50)
    for kw in kwargs:
        if kw in query_params:
            payload[kw] = kwargs[kw]
    # make the request
    r = requests.get(url, params=payload)
    if r.status_code != 200:
        raise Exception(f"Request failed with status code {r.status_code}.")
    return r

def format_time(time_string):
    '''
    Format the time representation to comply with the ChemRxiv format requirement.
    
    :param time_string: a string of time
    :type time_string: str
    
    :return: a string of time that complies with the ChemRxiv format requirement
    :rtype: str
    '''
    parsed_time = dateutil.parser.parse(time_string)
    return parsed_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]+'Z'
