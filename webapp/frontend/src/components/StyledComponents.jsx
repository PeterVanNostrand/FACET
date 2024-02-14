import { styled } from '@mui/material/styles';
import Button from '@mui/material/Button';
import Slider from '@mui/material/Slider';
import Switch from '@mui/material/Switch';

export const StyledIconButton = styled(Button)(({ theme }) => ({
    color: '#fff',
    width: '50px',
    height: '50px',
    '&:hover': {
        backgroundColor: '#f5f5f5',
        color: '#3c52b2',
    },
    // maxWidth: '100%',
    // maxHeight: '100%',
    // overflow: 'hidden',
}));

export const StyledSwitch = styled(Switch)(({ theme }) => ({
    size: 'medium',
    width: 60, 
    height: 36,
    color: '#6b8eff',
    size: 'large',
    '& .MuiSwitch-thumb': {
    },
    
    '& .MuiSwitch-track': {
        //borderRadius: 24 / 2, // Adjust border radius to make it round
        color: '#6b8eff',
        '&:before, &:after': {
            color: '#6b8eff',
            content: '""',
            position: 'absolute',
        },
        '&:before': {
            color: '#6b8eff',
            content: '"ON"', 
            left: 13,
            color: '#fff',
            fontSize: 10,
            fontWeight: 'bold',
        },
        '&:after': {
            content: '"OFF"',
            right: 12,
            color: '#fff',
            fontSize: 10,
            fontWeight: 'bold',
        },
    },
}));


export const StyledSlider = styled(Slider)(({ theme }) => ({
    color: '#6b8eff', 
    '& .MuiSlider-rail': {
        color: '#6b8eff',
    },
    '& .MuiSlider-thumb': {
        color: 'black',
        width: 5,
        height: 20,
        borderRadius: 5,
    },
    '& .MuiSlider-valueLabel': {
        background: '#6b8eff',
        borderRadius: 10,
    },
    '& .MuiSlider-mark': {
        backgroundColor: '#C8D2FB',
        width: 5,
        height: 20,
        borderRadius: 5,
    },
    '& .MuiSlider-markLabel': {
        fontSize: 15,
    },
}));


