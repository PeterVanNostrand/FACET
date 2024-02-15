import { styled } from '@mui/material/styles';
import Button from '@mui/material/Button';
import Slider from '@mui/material/Slider';
import Switch from '@mui/material/Switch';

export const StyledIconButton = styled(Button)(({ theme }) => ({
    color: '#fff',
    '&:hover': {
        backgroundColor: '#f5f5f5',
        color: '#3c52b2',
        width: '30px',
        height: '50px',
    },
    overflow: 'hidden',
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
            },
            '& .MuiSwitch-thumb': {
                backgroundColor: () => (props.checked ? '#6b8eff' : '#fff'), // Set thumb color based on checked state
            },
            '& .MuiSwitch-track': {
                // borderRadius: 24 / 2, // Adjust border radius to make it round
                '&:before, &:after': {
                    content: '""',
                    position: 'absolute',
                },
                '&:before': {
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
        }}
    />
);
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
    '& .MuiSlider-mark[data-index="1"]': {
        width: 15,
        height: 15,
        borderRadius: 10,
        backgroundColor: 'black',
        opacity: 1,
    },

}));


