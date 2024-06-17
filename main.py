try:
    print('Importing libraries')
    from bottle import route, run, request
    from ultralytics import YOLO
    import numpy as np
    import easyocr
    import string
    from PIL import Image#, ImageFilter
    import cv2
    from fuzzywuzzy import process
except Exception as e:
    print(f'Unable to load libraries with error : {e}')
    exit(1)

try:
    import requests
    import json
    
    print('Import complete')
except:
    print("problem importing requests, and json")


"""
Brief   :   Models are based off of YOLOv8, coco_model being the vanilla YOLOv8n and lisc_model being 
            custom trained YOLOv8n. 'n' represents the 'nano' version of the models, which has been chosen
            to reduce proessing load on the device.

Type    :   coco_model will be a pytorch model
            lisc_model will be a pytorch model

Improvements    :   Since coco_model has to process larger images, it too will be converted to OpenVINO format

"""
try:
    print("Loading models...")
    coco_model       = YOLO('yolov8n.pt', task='detect')
    license_model    = YOLO('best.pt',    task='detect')
    print('Loading complete')
except Exception as e:
    print(f'Unable to load models, exception : {e}')

'''

flag controlling use of VAHAN API to get vehicle data


'''

use_vahan = True



"""
/*
    The values of status is defined by the fields:
    1. VDS : Vehicle Detection Status, if 1, a vehicle was detected in the image
    2. LDS: License Plate Detection Status, if 1, a licenseplate was detected in the image
    3. CF : Compiles Format, if 1, the detected licenseplate complies with Indian format, no gurantee of corrent detection

    The status int will obtain values as per the following cases:

     -----------------------------------------------
    |   VDS     |   LDS     |    CF     |   status  | 
    |-----------|-----------|-----------|-----------|
    |   0       |   0       |   0       |   0       | 
    |   0       |   0       |   1       |   1       |
    |   0       |   1       |   0       |   2       |
    |   0       |   1       |   1       |   3       |
    |   1       |   0       |   0       |   4       |
    |   1       |   0       |   1       |   5       |
    |   1       |   1       |   0       |   6       |
    |   1       |   1       |   1       |   7       |      
    |-----------|-----------|-----------|-----------|


    confidence : Defines the average confidence in the detection of the licenseplate text. This is not very indicative of the final result 
                 of text detection. Some corrent deections show confidence of 0.2, while completely incorrect ones show it to be 0.97.

    license_plate : Will have the formatted licenseplate number. Is present iff CF is true. ie in cases status is : 1,3,5 or 7

    size         : int16, contains the size in bytes of image

    filename    :   Name of the image file, preferably stored as grayscale to reduce size. 

    vahan_response : Response obtained from VAHAN API

    vahan_size     : Size of Vahan Requests
     
    */
"""
send_dict = {}

"""
brief : result_dict, stores values of LDS, VDS, CF and any temporary variables required

fields: VDS, Vehicle Detection Status, boolean
        LDS, Licenseplate Detection Status, boolean
        CF, Complies Format, boolean
"""

result_dict = {}



####-- system variables

detection_classes   = [2,3,5,7]
##-- Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

##-- Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5',
                    'Z': '7',
                    'B': '8',
                    'Q': '0',
                    'P': '3',
                    'H': '4'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '2': 'Z',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    '7': 'Z',
                    '8': 'B'}

valid_license_plate_codes = [
'WB01',
'WB02',
'WB03',
'WB04',
'WB05',
'WB06',
'WB07',
'WB08',
'WB09',
'WB10',
'WB11',
'WB12',
'WB13',
'WB14',
'WB15',
'WB16',
'WB17',
'WB18',
'WB19',
'WB20',
'WB21',
'WB22',
'WB23',
'WB24',
'WB25',
'WB26',
'WB27',
'WB28',
'WB29',
'WB30',
'WB31',
'WB32',
'WB33',
'WB34',
'WB35',
'WB36',
'WB37',
'WB38',
'WB39',
'WB40',
'WB41',
'WB42',
'WB43',
'WB44',
'WB45',
'WB46',
'WB47',
'WB48',
'WB49',
'WB50',
'WB51',
'WB52',
'WB53',
'WB54',
'WB55',
'WB56',
'WB57',
'WB58',
'WB59',
'WB60',
'WB61',
'WB62',
'WB63',
'WB64',
'WB65',
'WB66',
'WB67',
'WB68',
'WB69',
'WB70',
'WB71',
'WB72',
'WB73',
'WB74',
'WB75',
'WB76',
'WB77',
'WB78',
'WB79',
'WB80',
'WB81',
'WB82',
'WB83',
'WB84',
'WB85',
'WB86',
'WB87',
'WB88',
'WB89',
'WB90',
'WB91',
'WB92',
'WB93',
'WB94',
'WB95',
'WB96',
'WB97',
'WB98',
'JH01',
'JH02',
'JH03',
'JH04',
'JH05',
'JH06',
'JH07',
'JH08',
'JH09',
'JH10',
'JH11',
'JH12',
'JH13',
'JH14',
'JH15',
'JH16',
'JH17',
'JH18',
'JH19',
'JH20',
'JH21',
'JH22',
'JH23',
'JH24',
'AP30',
'AP37',
'AP39',
'AN 01',
'AP40',
'AR01',
'AR02',
'AR03',
'AR04',
'AR05',
'AR06',
'AR07',
'AR08',
'AR09',
'AR10',
'AR11',
'AR12',
'AR13',
'AR14',
'AR15',
'AR16',
'AR17',
'AR19',
'AR20',
'AR22',
'AS01',
'AS02',
'AS03',
'AS04',
'AS05',
'AS06',
'AS07',
'AS08',
'AS09',
'AS10',
'AS11',
'AS12',
'AS13',
'AS14',
'AS15',
'AS16',
'AS17',
'AS18',
'AS19',
'AS20',
'AS21',
'AS22',
'AS23',
'AS24',
'AS25',
'AS26',
'AS27',
'AS29',
'AS30',
'AS31',
'AS32',
'AS33',
'AS34',
'BR01',
'BR02',
'BR03',
'BR04',
'BR05',
'BR06',
'BR07',
'BR08',
'BR09',
'BR10',
'BR11',
'BR19',
'BR21',
'BR22',
'BR24',
'BR25',
'BR26',
'BR27',
'BR28',
'BR29',
'BR30',
'BR31',
'BR32',
'BR33',
'BR34',
'BR37',
'BR38',
'BR39',
'BR43',
'BR44',
'BR45',
'BR46',
'BR50',
'BR51',
'BR52',
'BR53',
'BR55',
'BR56',
'CG01',
'CG02',
'CG03',
'CG04',
'CG05',
'CG06',
'CG07',
'CG08',
'CG09',
'CG10',
'CG11',
'CG12',
'CG13',
'CG14',
'CG15',
'CG16',
'CG17',
'CG18',
'CG19',
'CG20',
'CG21',
'CG22',
'CG23',
'CG24',
'CG25',
'CG26',
'CG27',
'CG28',
'CG29',
'CG30',
'CH01',
'CH02',
'CH03',
'CH04',
'DD01',
'DD02',
'DD03',
'DL1',
'DL2',
'DL3',
'DL4',
'DL5',
'DL6',
'DL7',
'DL8',
'DL9',
'DL10',
'DL11',
'DL12',
'DL13',
'GA01',
'GA02',
'GA03',
'GA04',
'GA05',
'GA06',
'GA07',
'GA08',
'GA09',
'GA10',
'GA11',
'GA12',
'GJ1',
'GJ2',
'GJ3',
'GJ4',
'GJ5',
'GJ6',
'GJ7',
'GJ8',
'GJ9',
'GJ10',
'GJ11',
'GJ12',
'GJ13',
'GJ14',
'GJ15',
'GJ16',
'GJ17',
'GJ18',
'GJ19',
'GJ20',
'GJ21',
'GJ22',
'GJ23',
'GJ24',
'GJ25',
'GJ26',
'GJ27',
'GJ28',
'GJ29',
'GJ30',
'GJ31',
'GJ32',
'GJ33',
'GJ34',
'GJ35',
'GJ36',
'GJ37',
'GJ38',
'GJ39',
'HP01',
'HP02',
'HP03',
'HP04',
'HP05',
'HP06',
'HP07',
'HP08',
'HP09',
'HP10',
'HP11',
'HP12',
'HP13',
'HP14',
'HP15',
'HP16',
'HP16AA',
'HP17',
'HP18',
'HP19',
'HP19AA',
'HP20',
'HP21',
'HP22',
'HP23',
'HP24',
'HP25',
'HP26',
'HP27',
'HP28',
'HP29',
'HP30',
'HP31',
'HP32',
'HP33',
'HP33AA',
'HP34',
'HP35',
'HP35AA',
'HP36',
'HP37',
'HP38',
'HP39',
'HP40',
'HP41',
'HP42',
'HP43',
'HP44',
'HP45',
'HP46',
'HP47',
'HP48',
'HP49',
'HP50',
'HP51 & HP52',
'HP53',
'HP54',
'HP55',
'HP56',
'HP57',
'HP58',
'HP59',
'HP60',
'HP61',
'HP62',
'HP63',
'HP64',
'HP65',
'HP66',
'HP67',
'HP68',
'HP69 & HP70',
'HP71',
'HP72',
'HP73',
'HP74',
'HP75',
'HP76',
'HP77',
'HP78',
'HP79',
'HP80',
'HP81',
'HP82',
'HP83',
'HP84',
'HP85',
'HP86',
'HP87',
'HP87AA',
'HP88',
'HP89',
'HP90',
'HP91',
'HP92',
'HP93',
'HP94',
'HP95',
'HP96',
'HP97',
'HP98',
'HP99',
'HR01',
'HR02',
'HR03',
'HR04',
'HR05',
'HR06',
'HR07',
'HR08',
'HR09',
'HR10',
'HR11',
'HR12',
'HR13',
'HR14',
'HR15',
'HR16',
'HR17',
'HR18',
'HR19',
'HR20',
'HR21',
'HR22',
'HR23',
'HR24',
'HR25',
'HR26',
'HR27',
'HR28',
'HR29',
'HR30',
'HR31',
'HR32',
'HR33',
'HR34',
'HR35',
'HR36',
'HR37',
'HR38',
'HR39',
'HR40',
'HR41',
'HR42',
'HR43',
'HR44',
'HR45',
'HR46',
'HR47',
'HR48',
'HR49',
'HR50',
'HR51',
'HR52',
'HR53',
'HR54',
'HR55',
'HR56',
'HR57',
'HR58',
'HR59',
'HR60',
'HR61',
'HR62',
'HR63',
'HR64',
'HR65',
'HR66',
'HR67',
'HR68',
'HR69',
'HR70',
'HR71',
'HR72',
'HR73',
'HR74',
'HR75',
'HR76',
'HR77',
'HR78',
'HR79',
'HR80',
'HR81',
'HR82',
'HR83',
'HR84',
'HR85',
'HR86',
'HR87',
'HR88',
'HR89',
'HR90',
'HR91',
'HR92',
'HR93',
'HR94',
'HR95',
'HR96',
'HR97',
'HR98',
'HR99',
'JK01',
'JK02',
'JK03',
'JK04',
'JK05',
'JK06',
'JK08',
'JK09',
'JK11',
'JK12',
'JK13',
'JK14',
'JK15',
'JK16',
'JK17',
'JK18',
'JK19',
'JK20',
'JK21',
'JK22',
'KA01',
'KA02',
'KA03',
'KA04',
'KA05',
'KA06',
'KA07',
'KA08',
'KA09',
'KA10',
'KA11',
'KA12',
'KA13',
'KA14',
'KA15',
'KA16',
'KA17',
'KA18',
'KA19',
'KA20',
'KA21',
'KA22',
'KA23',
'KA24',
'KA25',
'KA26',
'KA27',
'KA28',
'KA29',
'KA30',
'KA31',
'KA32',
'KA33',
'KA34',
'KA35',
'KA36',
'KA37',
'KA38',
'KA39',
'KA40',
'KA41',
'KA42',
'KA43',
'KA44',
'KA45',
'KA46',
'KA47',
'KA48',
'KA49',
'KA50',
'KA51',
'KA52',
'KA53',
'KA54',
'KA55',
'KA56',
'KA57',
'KA58',
'KA59',
'KA60',
'KA61',
'KA62',
'KA63',
'KA64',
'KA65',
'KA66',
'KA67',
'KA68',
'KA69',
'KA70',
'KA71',
'KL01',
'KL02',
'KL03',
'KL04',
'KL05',
'KL06',
'KL07',
'KL08',
'KL09',
'KL10',
'KL11',
'KL12',
'KL13',
'KL14',
'KL15',
'KL16',
'KL17',
'KL18',
'KL19',
'KL20',
'KL21',
'KL22',
'KL23',
'KL24',
'KL25',
'KL26',
'KL27',
'KL28',
'KL29',
'KL30',
'KL31',
'KL32',
'KL33',
'KL34',
'KL35',
'KL36',
'KL37',
'KL38',
'KL39',
'KL40',
'KL41',
'KL42',
'KL43',
'KL44',
'KL45',
'KL46',
'KL47',
'KL48',
'KL49',
'KL50',
'KL51',
'KL52',
'KL53',
'KL54',
'KL55',
'KL56',
'KL57',
'KL58',
'KL59',
'KL60',
'KL61',
'KL62',
'KL63',
'KL64',
'KL65',
'KL66',
'KL67',
'KL68',
'KL69',
'KL70',
'KL71',
'KL72',
'KL73',
'KL74',
'KL75',
'KL76',
'KL77',
'KL78',
'KL79',
'KL80',
'KL81',
'KL82',
'KL83',
'KL84',
'KL85',
'KL86',
'KL99',
'LA01',
'LA02',
'LD01',
'LD02',
'LD03',
'LD04',
'LD05',
'LD06',
'LD07',
'LD08',
'LD09',
'MH01',
'MH02',
'MH03',
'MH04',
'MH05',
'MH06',
'MH07',
'MH08',
'MH09',
'MH10',
'MH11',
'MH12',
'MH13',
'MH14',
'MH15',
'MH16',
'MH17',
'MH18',
'MH19',
'MH20',
'MH21',
'MH22',
'MH23',
'MH24',
'MH25',
'MH26',
'MH27',
'MH28',
'MH29',
'MH30',
'MH31',
'MH32',
'MH33',
'MH34',
'MH35',
'MH36',
'MH37',
'MH38',
'MH39',
'MH40',
'MH41',
'MH42',
'MH43',
'MH44',
'MH45',
'MH46',
'MH47',
'MH48',
'MH49',
'MH50',
'MH51',
'MH52',
'MH53',
'MH54',
'ML01',
'ML02',
'ML04',
'ML05',
'ML06',
'ML07',
'ML08',
'ML09',
'ML10',
'MN01',
'MN02',
'MN03',
'MN04',
'MN05',
'MN06',
'MN07',
'MP01',
'MP02',
'MP03',
'MP04',
'MP05',
'MP06',
'MP07',
'MP08',
'MP09',
'MP10',
'MP11',
'MP12',
'MP13',
'MP14',
'MP15',
'MP16',
'MP17',
'MP18',
'MP19',
'MP20',
'MP21',
'MP22',
'MP28',
'MP30',
'MP31',
'MP32',
'MP33',
'MP34',
'MP35',
'MP36',
'MP37',
'MP38',
'MP39',
'MP40',
'MP41',
'MP42',
'MP43',
'MP44',
'MP45',
'MP46',
'MP47',
'MP48',
'MP49',
'MP50',
'MP52',
'MP52',
'MP53',
'MP54',
'MP65',
'MP66',
'MP67',
'MP68',
'MP69',
'MP70',
'MP71',
'MZ01',
'MZ02',
'MZ03',
'MZ04',
'MZ05',
'MZ06',
'MZ07',
'MZ08',
'NL01',
'NL02',
'NL03',
'NL04',
'NL05',
'NL06',
'NL07',
'NL08',
'NL09',
'NL10',
'OD01',
'OD02',
'OD03',
'OD04',
'OD05',
'OD06',
'OD07',
'OD08',
'OD09',
'OD10',
'OD11',
'OD12',
'OD13',
'OD14',
'OD15',
'OD16',
'OD17',
'OD18',
'OD19',
'OD20',
'OD21',
'OD22',
'OD23',
'OD24',
'OD25',
'OD26',
'OD27',
'OD28',
'OD29',
'OD30',
'OD31',
'OD32',
'OD33',
'OD34',
'OD35',
'PB01',
'PB02',
'PB03',
'PB04',
'PB05',
'PB06',
'PB07',
'PB08',
'PB09',
'PB10',
'PB11',
'PB12',
'PB13',
'PB14',
'PB15',
'PB16',
'PB17',
'PB18',
'PB19',
'PB20',
'PB21',
'PB22',
'PB23',
'PB24',
'PB25',
'PB26',
'PB27',
'PB28',
'PB29',
'PB30',
'PB31',
'PB32',
'PB33',
'PB34',
'PB35',
'PB36',
'PB37',
'PB38',
'PB39',
'PB40',
'PB41',
'PB42',
'PB43',
'PB44',
'PB45',
'PB46',
'PB47',
'PB48',
'PB49',
'PB50',
'PB51',
'PB52',
'PB53',
'PB54',
'PB55',
'PB56',
'PB57',
'PB58',
'PB59',
'PB60',
'PB61',
'PB62',
'PB63',
'PB64',
'PB65',
'PB66',
'PB67',
'PB68',
'PB69',
'PB70',
'PB71',
'PB72',
'PB73',
'PB74',
'PB75',
'PB76',
'PB77',
'PB78',
'PB79',
'PB80',
'PB81',
'PB82',
'PB83',
'PB84',
'PB85',
'PB86',
'PB87',
'PB88',
'PB89',
'PB90',
'PB91',
'PB92',
'PB99',
'PY01',
'PY02',
'PY03',
'PY04',
'PY05',
'RJ01',
'RJ02',
'RJ03',
'RJ04',
'RJ05',
'RJ06',
'RJ07',
'RJ08',
'RJ09',
'RJ10',
'RJ11',
'RJ12',
'RJ13',
'RJ14',
'RJ15',
'RJ16',
'RJ17',
'RJ18',
'RJ19',
'RJ20',
'RJ21',
'RJ22',
'RJ23',
'RJ24',
'RJ25',
'RJ26',
'RJ27',
'RJ28',
'RJ29',
'RJ30',
'RJ31',
'RJ32',
'RJ33',
'RJ34',
'RJ35',
'RJ36',
'RJ37',
'RJ38',
'RJ39',
'RJ40',
'RJ41',
'RJ42',
'RJ43',
'RJ44',
'RJ45',
'RJ46',
'RJ47',
'RJ48',
'RJ49',
'RJ50',
'RJ51',
'RJ52',
'RJ53',
'RJ54',
'RJ55',
'RJ56',
'RJ57',
'RJ58',
'SK01',
'SK02',
'SK03',
'SK04',
'SK05',
'SK06',
'SK07',
'SK08',
'TN01',
'TN02',
'TN03',
'TN04',
'TN05',
'TN06',
'TN07',
'TN09',
'TN10',
'TN11',
'TN12',
'TN13',
'TN14',
'TN15',
'TN15M',
'TN16',
'TN16Z',
'TN18',
'TN18Y',
'TN19',
'TN19Y',
'TN19Z',
'TN20',
'TN20X',
'TN21',
'TN22',
'TN23',
'TN23T',
'TN24',
'TN25',
'TN27',
'TN28',
'TN28Z',
'TN29',
'TN29W',
'TN29Z',
'TN30',
'TN30W',
'TN31',
'TN31Y',
'TN31Z',
'TN32',
'TN33',
'TN34',
'TN34M',
'TN36',
'TN36W',
'TN36Z',
'TN37',
'TN37Z',
'TN38',
'TN39',
'TN39Z',
'TN40',
'TN41',
'TN41W',
'TN42',
'TN42Y',
'TN43',
'TN43Z',
'TN45',
'TN45Z',
'TN46',
'TN47',
'TN47X',
'TN47Y',
'TN47Z',
'TN48',
'TN48X',
'TN48Y',
'TN48Z',
'TN49',
'TN49Y',
'TN50',
'TN50Y',
'TN50Z',
'TN51',
'TN52',
'TN54',
'TN55',
'TN55BQ',
'TN55Y',
'TN55Z',
'TN56',
'TN57',
'TN57W',
'TN57V',
'TN57W',
'TN58',
'TN58Y',
'TN58Z',
'TN59',
'TN59V',
'TN59Z',
'TN60',
'TN60Z',
'TN61',
'TN63',
'TN63Z',
'TN64',
'TN65',
'TN65Z',
'TN66',
'TN67',
'TN67W',
'TN68',
'TN69',
'TN70',
'TN72',
'TN72V',
'TN73',
'TN73Z',
'TN74',
'TN75',
'TN76',
'TN76Y',
'TN77',
'TN77Z',
'TN78',
'TN78M',
'TN79',
'TN81',
'TN81Z',
'TN82',
'TN82Z',
'TN83',
'TN83M',
'TN83Y',
'TN84',
'TN84U',
'TN85',
'TN86',
'TN87',
'TN88',
'TN88Z',
'TN90',
'TN91',
'TN91Z',
'TN92',
'TN93',
'TN94',
'TN94Z',
'TN95',
'TN96',
'TN97',
'TN97Z',
'TN99',
'TR01',
'TR02',
'TR03',
'TR04',
'TR05',
'TR06',
'TR07',
'TR08',
'TG01',
'TG02',
'TG03',
'TG04',
'TG05',
'TG06',
'TG07',
'TG08',
'TG09',
'TG10',
'TG11',
'TG12',
'TG13',
'TG14',
'TG15',
'TG16',
'TG17',
'TG18',
'TG19',
'TG20',
'TG21',
'TG22',
'TG23',
'TG24',
'TG25',
'TG26',
'TG27',
'TG28',
'TG29',
'TG30',
'TG31',
'TG32',
'TG33',
'TG34',
'TG35',
'TG36',
'TG37',
'TG38',
'UK01',
'UK02',
'UK03',
'UK04',
'UK05',
'UK06',
'UK07',
'UK08',
'UK09',
'UK10',
'UK11',
'UK12',
'UK13',
'UK14',
'UK15',
'UK16',
'UK17',
'UK18',
'UK19',
'UK20',
'UP1',
'UP2',
'UP3',
'UP4',
'UP5',
'UP6',
'UP7',
'UP8',
'UP9',
'UP10',
'UP11',
'UP12',
'UP13',
'UP14',
'UP15',
'UP16',
'UP17',
'UP19',
'UP20',
'UP21',
'UP22',
'UP23',
'UP24',
'UP25',
'UP26',
'UP27',
'UP30',
'UP31',
'UP32',
'UP33',
'UP34',
'UP35',
'UP36',
'UP37',
'UP38',
'UP40',
'UP41',
'UP42',
'UP43',
'UP44',
'UP45',
'UP46',
'UP47',
'UP50',
'UP51',
'UP52',
'UP53',
'UP54',
'UP55',
'UP56',
'UP57',
'UP58',
'UP60',
'UP61',
'UP62',
'UP63',
'UP64',
'UP65',
'UP66',
'UP67',
'UP70',
'UP71',
'UP72',
'UP73',
'UP74',
'UP75',
'UP76',
'UP77',
'UP78',
'UP79',
'UP80',
'UP81',
'UP82',
'UP83',
'UP84',
'UP85',
'UP86',
'UP87',
'UP90',
'UP91',
'UP92',
'UP93',
'UP94',
'UP95',
'UP96'
]

##---------------------------------------------------------------------------------------------------##
##--                                    Helper Functions                                           --##
##---------------------------------------------------------------------------------------------------##
def license_complies_format(text):
    """
    Check if the license plate text complies with the required format using NumPy.

    Args:
        text (str): License plate text.
        dict_char_to_int (dict): Dictionary mapping characters to integers.
        dict_int_to_char (dict): Dictionary mapping integers to characters.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    # Create an array from the text
    text_array = np.array(list(text), dtype='<U1')
    print('text_array : ', text_array)
    length = len(text)
    checks = []
    # Define the checks for each position
    if length == 10:
        checks = [
            np.isin(text_array[0:2], list(string.ascii_uppercase)),  # State code
            np.isin(text_array[2:4], list('0123456789')),  # District code
            np.isin(text_array[4:6], list(string.ascii_uppercase)),  # Random letters
            np.isin(text_array[6:8], list('0123456789')),  # Random numbers
        ]
    elif length == 9:
        checks = [
                np.isin(text_array[0:2], list(string.ascii_uppercase)),  # State code
                np.isin(text_array[2:4], list('0123456789')),  # District code
                np.isin(text_array[4:5], list(string.ascii_uppercase)),  # Random letters
                np.isin(text_array[5:], list('0123456789'))  # Random numbers
            ]
    else :
        print('License Text is not of lenght 10 or 9')
        return False

    # Check if all conditions are met
    try:
        return all(np.all(check) for check in checks)
    except Exception as e:
        print("An error occured in license_complies_format() while performing checks : ", e)
        return False

def closest_match(target, possibilities):
    result = process.extract(target, possibilities, limit=4)
    return result[0],result  # Returns the closest match

def make_post_request(payload):
    url = "https://api.attestr.com/api/v1/public/checkx/rc"
    headers = {
        'Authorization': 'Basic T1gwbjZ6TEJ6Z1VOYmdXRzRnLjY5YzNlMDE2YmY0OTk0NWVkMDA4NjgzNWZjOTBiNjBkOmY5YTU1ZDM4Y2ZhZDcyMzJlODMxMDhmMTE4NWEwMzg4MzNkZTBmNzdiMzY2NTk1NA==',
        'Content-Type': 'application/json'
    }
    payload = {
        "reg": payload
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            print(response.text)
            return response.text
        else:
            # If the request failed, raise an exception with the status code
            response.raise_for_status()
    except requests.exceptions.RequestException as e:
        # Handle any exceptions that occur during the request
        print("Error:", e)
        return None
    
def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries. 
    The mapping dictionaries are used to convert common errors such as S <-> 5 , and 4 <-> A
    since we know where we can expect characters and numbers in our numberplates.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
             None, if len(text) exceeds 10, but this is redundant since a check is already being performed
    """

    length = len(text)
    license_plate_ = ''


    if length==10:
        mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
        mapping = {
                    0: dict_int_to_char,
                    1: dict_int_to_char,
                    2: dict_char_to_int,
                    3: dict_char_to_int,
                    4: dict_int_to_char,
                    5: dict_int_to_char,
                    6: dict_char_to_int,
                    7: dict_char_to_int,
                    8: dict_char_to_int,
                    9: dict_char_to_int
                   }
        for j in range(10):
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]

    elif length==9:
        mapping = {
                    0: dict_int_to_char,
                    1: dict_int_to_char,
                    2: dict_char_to_int,
                    3: dict_char_to_int,
                    4: dict_int_to_char,
                    5: dict_char_to_int,
                    6: dict_char_to_int,
                    7: dict_char_to_int,
                    8: dict_char_to_int
                   }
        for j in range(9):
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]
    else:
        print("Length is : ",len(text), "text being : ", text)
        return text
    
    return license_plate_

def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """
    distance,angle = Hough_lines(license_plate_crop)
    if angle is not None:
        print("Angle is :", angle)
        license_plate_crop = rotateImage(license_plate_crop,angle)
    else:
        pass

    
    cv2.imshow('lp rot', license_plate_crop)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    if license_plate_crop.shape[0] > 70:
        detections = reader.readtext(license_plate_crop, decoder='greedy', detail=1, blocklist="~`?\\\'\";?,|=+-_)(*&^%$#@!).-]{[}'" , contrast_ths = 0.5, mag_ratio = 2, add_margin=0.0)
    else:
        detections = reader.readtext(license_plate_crop, decoder='greedy', detail=1, blocklist="~`?\\\'\";?,|=+-_)(*&^%$#@!).-]{[}'" , contrast_ths = 0.5, mag_ratio = 1, add_margin=0.0)

    if not detections:
        print("No detection")
        return None, None
    lp_text = ""
    i = 0
    total_conf = 0
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        lp_text += text     
        i+=1
        total_conf+=score  


    print("Sending text for formatting : ", lp_text)
    lp_text = format_license(lp_text)

    rto_code, results = closest_match(str(lp_text[:4]),valid_license_plate_codes)
    print("Possible codes are : ", results)
    lp_text = rto_code[0] + lp_text[4:]


    if license_complies_format(lp_text):
        return lp_text, total_conf/i
    else:
        return lp_text, 2

def license_result_process(license_results,
                           image_for_lpd,
                           license_plate_text = [],
                           license_plate_text_score=[],
                           random_text = [],
                           time = "default"
                           ):
    
    image_for_lpd  = np.asarray(image_for_lpd)
    
    for result in license_results: ##-- For each image passed
        ##--debug
        cv2.imshow("detection", result.plot())
        cv2.waitKey(1000)
        for license_plate in result.boxes.data.tolist():
            
            x1, y1, x2, y2, score, class_id = license_plate
            image_for_ocr = image_for_lpd[int(y1):int(y2),int(x1):int(x2),:]##-- This confusing arrangement is caused by the different standards used by ultralytics and numpy in deciding x,y coordinate orientation

            cv2.imshow('Image',image_for_ocr)
            cv2.waitKey(1000)
            
            ##--debug
            # cv2.imwrite("lp_image.jpg",image_for_ocr) 
            
            # image_for_ocr = cv2.cvtColor(image_for_ocr, cv2.COLOR_BGR2GRAY) ##-- Convert the color to grayscale
            image_for_ocr = np.dot(image_for_ocr[...,:3],[0.2989,0.5870,0.1140]) ##-- Convert the color to grayscale

            cv2.imwrite(time+".jpg",image_for_ocr)
            # assert len(image_for_ocr.shape)==2, "Image not of correct dimensions, requires image to be 2D"
            send_dict['size'] = image_for_ocr.shape[0] * image_for_ocr.shape[1]
            send_dict['filename'] = time+".jpg"


            min_val = np.min(image_for_ocr)
            max_val = np.max(image_for_ocr)
            image_for_ocr = ((image_for_ocr-min_val)/(max_val-min_val))*255
            image_for_ocr = image_for_ocr.astype(np.uint8)

            image_for_ocr = cv2.resize(image_for_ocr,(0,0), fx = 4, fy = 4)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
            image_for_ocr = clahe.apply(image_for_ocr)  


            image_for_ocr = cv2.convertScaleAbs(image_for_ocr,alpha = 3, beta = -127)

            # min_val = np.min(image_for_ocr)
            # max_val = np.max(image_for_ocr)

            # ##-- Increasing Contrast
            # print('Min_Val : ', min_val, 'Max_Val : ',max_val)
            # image_for_ocr = ((image_for_ocr-min_val)/(max_val-min_val))*200

            cv2.imshow('Image',image_for_ocr)
            cv2.waitKey(1000)
            # image_for_ocr = cv2.resize(image_for_ocr,(0,0), fx=2, fy=2)
            # image_for_ocr = cv2.bilateralFilter(image_for_ocr.astype(np.uint8),9,75,75)

            ##-- Applying threshold
            # threshold = 50
            # image_for_ocr = np.where(image_for_ocr > threshold, 255, 0)

            ##--debug
            # image_for_ocr = np.clip(image_for_ocr, 0, 255).astype(np.uint8)

            # cv2.imshow('Image',image_for_ocr)
            # cv2.waitKey(1000)
            # image_Canny = cv2.Canny(image_for_ocr,100,255)
            # cv2.imshow("Canny", image_Canny)
            # cv2.waitKey(1000)

            # image_for_ocr = np.clip(image_for_ocr, 0, 255).astype(np.uint8)
            # img_save = Image.fromarray(image_for_ocr)
            # img_save.save("DetectedLP.jpg")
            
            ##-- perform OCR on the Image
            """
            The execution order here is as follows:
                1. easyocr is used to perform image-to-text
                2. text is formatted(1), by removing spaces and '.'
                3. a check is performed to see if the licenseplate complies with the Indian format
                4. text is formatted(2), by replacing misdetected characters

            The detected, checked, formatted text is returned along with the confidence score of detection

            """

            text_result, conf = read_license_plate(image_for_ocr)
            if conf is not None:
                if conf>=1:
                    random_text.append(text_result)
                else:
                    license_plate_text.append(text_result)
                    license_plate_text_score.append(conf)
    ##--debug
    cv2.destroyAllWindows()
    return license_plate_text, license_plate_text_score, random_text

def Hough_lines(image: np.array):
    """
    brief   :   Finds longest straight line in the image using the values of pixels to contruct lines and finding max. 
                Computationally heavy and reserved for small images only.

    inputs  :   image, a numpy array.
                
    returns :   if lines are found in the image:
                    [distance,  = the distance of the longest line from the origin, ie. pixel (0,0)
                    angle]     = the angle in radian of the line with the vertical
                else:
                    None,None
    
    """
    assert len(image.shape) == 2, 'Image is not grayscale'
    image = cv2.Canny(image,100,200, apertureSize=3)
    cv2.imshow('Canny', image)
    cv2.waitKey(1000)
    lines = cv2.HoughLines(image,1,np.pi/180, 50)
    if lines is not None:
        max_len = 0
        distance = 0
        angle = 0
        ##-- Loop over the lines
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > max_len and theta > np.pi/4:
                max_len = length
                distance = rho
                angle = theta
        return distance,angle
    else:
        print("No lines found. The possible reasons are :\n 1. There is no straight line in the image.\n 2. The image is too low in resolution.")
        return None, None
    
def rotateImage(cvImage, rotate_angle: float):
    """
    brief   :   rotates an input image with the angle given in the arguments, with fulcrum at centre

    inputs  :   cvImage, the image that needs to be rotated
                angle,  the angle by which the image need sto be rotated. See OpenCV referece to determin the sing of the angle for desired rotation

    returns :   rotated image, with extrapolated borders. The extrapolation can be changed. Look at cv2.warpAffine() docs.

    """
    rotate_angle = rotate_angle*180/np.pi - 90
    if rotate_angle > 360 :
        rotate_angle = rotate_angle - 360
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage


##----------------------------------------------------------------------------------------------------


##-- Exposed API 
"""
Brief   :   Exposed Functions that can be called via an HTTP call to localhost:5050/predict, and can be passed a JSON object

Input   :   The acceptable input is a JSON object that has the following information:
            1. image_path, path to the image on which the predictions are to be performed
            2. time, time of the day, format to be decided soon

Example :   curl -X POST -H "Content-Type: application/json" -d "@data.json" http://localhost:5050/predict > output.json,
            where data.json is a local file that has the data in the format stored as stated above. Example being
            {
                "image_path":"site_images\\15.jpg",
                "time":0
            }      

Returns :   The JSON object has the fields as described above

Example :   {"VDS": true, "LDS": false, "LPC": null, "LPT": null, "CF": false, "AT": [[]]}

"""

@route('/predict', method='POST')    ##-- python bottle route decorator that exposes the function to HTTP requests
def predict():
    data  = request.json
    result_dict = {}
    detections = []
    random_text = []
    license_results = []
    license_plate_text = []
    license_plate_text_score = []
    # image = cv2.imread(data['image_path'])
    image_pil = Image.open(data['image_path'])
    image = np.array(image_pil)
    ##-- Add time based functionality here, includeing pre-processing image during night

    detections = coco_model.predict(image, classes = detection_classes, verbose = False)[0]

    if len(detections) == 0:
        ##-- No vehicle was detected by the coco_model, further preprocess the image
        result_dict['VDS'] = False
        ##-- Implement adaptive histogram equalization, i.e. increase contrast
        # image = cv2.equalizeHist(image) ##-- Just simple histogram implementation
        ##-- Implement gamma correction, i.e. adapt to brightness changes
        ##-- predict again
        detections = coco_model.predict(image, classes = detection_classes, verbose = False)
        if len(detections) == 0:
            result_dict['VDS'] = False
            ##-- Since vehicle detection has failed twice, try to find Liscenceplate using lisc_model
            license_results = license_model(image, verbose=False)[0]
    else:
        result_dict['VDS'] = True

    if not license_results: ##-- If license plate detections have not been performed, license_results is empty

        for detection in detections.boxes.data.tolist():            ##-- For each detected vehicle 
            x1, y1, x2, y2, score, class_id = detection             ##-- separate the detections
            image_for_lpd = image[int(y1):int(y2),int(x1):int(x2),:]##-- Crop image

            license_results = license_model(image_for_lpd,verbose=False)       ##-- detect numberplates

            license_plate_text, license_plate_text_score, random_text = license_result_process(license_results, image_for_lpd,license_plate_text, license_plate_text_score, random_text, data['time'])

    else:  ##-- If license plate detections have been performed, license_results is not empty
        license_plate_text, license_plate_text_score, random_text = license_result_process(license_results, image,license_plate_text, license_plate_text_score, random_text, data['time'])
        
    ##-- Out of detection loops
    if not license_plate_text_score or not license_plate_text:    ##-- If the license_plate_list is empty
        result_dict['LDS'] = False
        send_dict['confidence'] = 0
        send_dict['license_plate'] = ""
        result_dict['CF']  = False
        result_dict['AT']  = random_text

        VDS, LDS, CF = int(result_dict['VDS']), int(result_dict['LDS']), int(result_dict['CF'])
        send_dict['status'] = (VDS<<2)|(LDS<<1)|(CF)
        result_dict['AT'].append(license_plate_text)


        send_dict['vahan_response'] = ""
        send_dict['vahan_size']     = 0
    else:
        result_dict['LDS'] = True
        result_dict['CF']  = False if np.max(license_plate_text_score)>=1 else True

        send_dict['confidence'] = np.max(license_plate_text_score)
        send_dict['license_plate'] = license_plate_text[np.argmax(license_plate_text_score)]

        VDS, LDS, CF = int(result_dict['VDS']), int(result_dict['LDS']), int(result_dict['CF'])
        send_dict['status'] = (VDS<<2)|(LDS<<1)|(CF)

        ##-- Call the vahan API and 
        if use_vahan:
            vahan_response = make_post_request(send_dict['license_plate'])
            if vahan_response is not None:
                send_dict['vahan_response'] = vahan_response
                send_dict['vahan_size'] = len(vahan_response)
            else:
                send_dict['vahan_response'] = "No Response"
                send_dict['vahan_size'] = 0
        else:
            send_dict['vahan_response'] = ""
            send_dict['vahan_size']     = 0
    return send_dict


@route('/')
def return_true():
    return "Server Works, use by sending request to /predict"

@route('/hello') 
def hello():
    return "Hello World!"

run(host = 'localhost', port=5050, debug=True)


    
