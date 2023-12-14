import axios from 'axios';

export const fetchApplications = async () => {
    try {
        const response = await axios.get('http://localhost:3001/facet/applications');
        return response.data;
    } catch (error) {
        console.error(error);
        return [];
    }
};

export const fetchExplanations = async (selectedApplication, numExplanations, constraints) => {
    if (constraints.length === 0 || selectedApplication.length === 0) return [];

    try {
        const response = await axios.post(
            'http://localhost:3001/facet/explanations',
            {
                num_explanations: numExplanations,
                x0: selectedApplication.x0,
                x1: selectedApplication.x1,
                x2: selectedApplication.x2,
                x3: selectedApplication.x3,
                constraints: constraints,
            }
        );
        return response.data;
    } catch (error) {
        console.error(error);
        return [];
    }
};
