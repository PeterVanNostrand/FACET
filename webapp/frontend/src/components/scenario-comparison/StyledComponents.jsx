import { styled } from '@mui/material/styles';
import Slider from '@mui/material/Slider';

/* 
    MUI Slider styled to match number line from explanations, 
*/ 
export const StyledExplanationSlider = styled(Slider)(({ theme }) => ({
    color: '#006eff',
    overflow: 'off',
    '&.No_Change': { // In use when no change is needed (current value falls within accepted range )
        ...({
            filter: 'grayscale(100%)', // Apply grayscale filter 
        }),
    },
    '&.Mui-disabled': { // Override disable color
        color: '#006eff',
        pointerEvents: 'none', 
    },
    '& .MuiSlider-rail': {
        color: 'black',
        opacity: '1',
    },
    '& .MuiSlider-track': {
        height: 10,
         borderWidth: '2px',
    },
    '& .MuiSlider-thumb': {
        color: 'black',
        width: 3,
        height: 20,
        borderRadius: 5,
    },
    '& .MuiSlider-valueLabel': { // Handle/Thumb Labels:
        background: '#6b8eff',
        borderRadius: 10,
    },
    '& .MuiSlider-mark': { // Min and Max
        width: 3,
        height: 20,
        borderRadius: 5,
        backgroundColor: 'black',
        boxShadow: '0px 2px 4px rgba(0, 0, 0, 0.25)',
    },
    '& .MuiSlider-markLabel': { // Labels 
        fontSize: 15,
        fontWeight: 'bold',
        marginRight: '-20px',
    },
    '& .MuiSlider-mark[data-index="1"]': { // Current Value 
        width: 15,
        height: 15,
        borderRadius: 10,
        backgroundColor: 'red',
        opacity: 1,
    },
    '& .MuiSlider-markLabel[data-index="2"]': { // min range handle
        color: '#006eff',
        top: '-20px', 
        marginRight: '40px !important',

    },
    '& .MuiSlider-markLabel[data-index="3"]': { // max range handle
        color: '#006eff',
        top: '-20px', 
        marginLeft: '40px !important',
    },
}));


