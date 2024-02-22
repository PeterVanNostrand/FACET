import { styled } from '@mui/material/styles';
import Button from '@mui/material/Button';
import Slider from '@mui/material/Slider';
import Switch from '@mui/material/Switch';
import Avatar from '@mui/material/Avatar';

export const StyledIconButton = styled(Button)(({ theme }) => ({
    borderRadius: '5px',
    height: '40px',
    minWidth: '40px',
    padding: '0',
    '&:hover': {
        backgroundColor: '#f5f5f5',
        borderRadius: '5px',
        minHeight: '20px',
        minWidth: '20px',
        padding: '0',
    },
    '& .MuiTouchRipple-root': {
        width: '40px',
        height: '40px',
        transitionDuration: '0.5s',
        padding: '0',

    },
}));

export const StyledAvatar = styled(Avatar)(({ theme }) => ({
    width: '40px',
    height: '40px',
}));

export const StyledSwitch = (props) => (
    <Switch
        {...props}
        sx={{
            width: 62,
            height: 36,
            '& .MuiSwitch-switchBase.Mui-checked': {
                color: '#6b8eff',
                '&:hover': {
                    backgroundColor: 'rgba(107, 142, 255, 0.5)',
                },
            },
            '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                backgroundColor: '#6b8eff',
                color: '#6b8eff',
            },
            '& .MuiSwitch-thumb': {
                backgroundColor: () => (props.checked ? '#6b8eff' : '#fff'),
            },
            '& .MuiSwitch-track': {
                //borderRadius: 24 / 2, // Adjust border radius to make it round
                '&:before, &:after': {
                    content: '""',
                    position: 'absolute',
                },
                '&:before': {
                    content: '"ON"',
                    color: '#fff',
                    fontSize: 9,
                    fontWeight: 'bold',
                },
                '&:after': {
                    content: '"OFF"',
                    right: 15,
                    color: '#fff',
                    fontSize: 9,
                    fontWeight: 'bold',
                },
            },
        }}
    />
);
export const StyledSlider = styled(Slider)(({ theme }) => ({
    color: '#6b8eff',
    overflow: 'off',
    '& .MuiSlider-rail': {
        color: 'black',
    },
    '& .MuiSlider-track': {
        height: 10,
        borderRadius: 4,
    },
    '& .MuiSlider-thumb': {
        color: 'black',
        width: 5,
        height: 20,
        borderRadius: 5,
    },
    '& .MuiSlider-valueLabel': { // Handle/Thumb Labels:
        background: '#6b8eff',
        borderRadius: 10,
    },
    '& .MuiSlider-mark': { // Min and Max
        width: 5,
        height: 20,
        borderRadius: 5,
        backgroundColor: 'black',
        boxShadow: '0px 2px 4px rgba(0, 0, 0, 0.25)',
    },
    '& .MuiSlider-markLabel': { // Labels 
        fontSize: 15,
    },
    '& .MuiSlider-mark[data-index="1"]': { // Current Value 
        width: 15,
        height: 15,
        borderRadius: 10,
        backgroundColor: 'black',
        opacity: 1,
    },
}));


