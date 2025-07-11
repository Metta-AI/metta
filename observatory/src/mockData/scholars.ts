export interface Scholar {
    id: string;
    name: string;
    username: string;
    avatar: string;
    institution: string;
    department: string;
    title: string;
    bio: string;
    expertise: string[];
    hIndex: number;
    totalCitations: number;
    papers: Array<{
        id: string;
        title: string;
        year: number;
        citations: number;
        url: string;
    }>;
    recentActivity: string;
    isFollowing: boolean;
    claimed: boolean;
}

// Helper function to generate random data
const getRandomElement = <T>(array: T[]): T => array[Math.floor(Math.random() * array.length)];
const getRandomElements = <T>(array: T[], count: number): T[] => {
    const shuffled = [...array].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
};

const institutions = [
    'Stanford University', 'MIT', 'UC Berkeley', 'CMU', 'Harvard University',
    'University of Toronto', 'University of Oxford', 'University of Cambridge',
    'ETH Zurich', 'EPFL', 'Imperial College London', 'UCL', 'University of Edinburgh',
    'University of Manchester', 'University of Bristol', 'University of Warwick',
    'University of Sheffield', 'University of Leeds', 'University of Liverpool',
    'University of Nottingham', 'University of Birmingham', 'Cardiff University',
    'University of Glasgow', 'University of Dundee', 'University of Aberdeen',
    'University of Stirling', 'University of Strathclyde', 'Heriot-Watt University'
];

const departments = [
    'Computer Science', 'Electrical Engineering & Computer Science', 'Mathematics',
    'Machine Learning', 'AI Research', 'Robotics', 'Computer Vision',
    'Natural Language Processing', 'Theoretical Computer Science', 'Statistics',
    'Information Science', 'Data Science', 'Computational Biology'
];

const titles = [
    'Assistant Professor', 'Associate Professor', 'Professor', 'Research Scientist',
    'Senior Research Scientist', 'Principal Research Scientist', 'Research Engineer',
    'Senior Research Engineer', 'AI Researcher', 'Machine Learning Engineer'
];

const expertiseAreas = [
    'Attention Mechanisms', 'Transformer Architecture', 'Natural Language Processing',
    'Reinforcement Learning', 'Robotics', 'Control Theory', 'Neural Network Theory',
    'Approximation Algorithms', 'Theoretical Computer Science', 'Computer Vision',
    'Deep Learning', 'Convolutional Neural Networks', 'Generative Models',
    'Variational Methods', 'Large Language Models', 'Computer Graphics',
    'Machine Learning', 'AI Safety', 'Quantum Computing', 'Systems',
    'Audio AI', 'Recommendation Systems', 'Autonomous Vehicles', 'AR/VR',
    'Conversational AI', 'Document AI', 'Code AI', 'Question Answering'
];

const firstNames = [
    'Alice', 'Bob', 'Carol', 'David', 'Eva', 'Frank', 'Grace', 'Henry', 'Iris', 'Jack',
    'Kate', 'Liam', 'Maya', 'Noah', 'Olivia', 'Paul', 'Quinn', 'Ruby', 'Sam', 'Tara',
    'Uma', 'Victor', 'Wendy', 'Xavier', 'Yara', 'Zoe', 'Alex', 'Blake', 'Casey', 'Drew',
    'Emery', 'Finley', 'Gray', 'Harper', 'Indigo', 'Jordan', 'Kendall', 'Logan', 'Morgan', 'Nova',
    'Ocean', 'Parker', 'Quincy', 'River', 'Sage', 'Taylor', 'Unity', 'Valor', 'Winter', 'Xander',
    'Aria', 'Bennett', 'Clara', 'Dexter', 'Eliza', 'Felix', 'Gemma', 'Hugo', 'Ivy', 'Jasper',
    'Kiera', 'Leo', 'Mila', 'Nash', 'Opal', 'Phoenix', 'Quinn', 'Raven', 'Silas', 'Thea',
    'Uriah', 'Violet', 'Wyatt', 'Xena', 'Yale', 'Zara', 'Atlas', 'Briar', 'Cedar', 'Dove',
    'Echo', 'Flint', 'Grove', 'Haven', 'Indigo', 'Jade', 'Kai', 'Luna', 'Moss', 'Nova',
    'Orion', 'Pine', 'Quill', 'Ridge', 'Sage', 'Thorn', 'Umber', 'Vale', 'Wren', 'Xero'
];

const lastNames = [
    'Johnson', 'Chen', 'Williams', 'Kumar', 'Rodriguez', 'Smith', 'Brown', 'Davis', 'Miller', 'Wilson',
    'Moore', 'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin', 'Thompson', 'Garcia',
    'Martinez', 'Robinson', 'Clark', 'Rodriguez', 'Lewis', 'Lee', 'Walker', 'Hall', 'Allen', 'Young',
    'King', 'Wright', 'Lopez', 'Hill', 'Scott', 'Green', 'Adams', 'Baker', 'Gonzalez', 'Nelson',
    'Carter', 'Mitchell', 'Perez', 'Roberts', 'Turner', 'Phillips', 'Campbell', 'Parker', 'Evans', 'Edwards',
    'Collins', 'Stewart', 'Sanchez', 'Morris', 'Rogers', 'Reed', 'Cook', 'Morgan', 'Bell', 'Murphy',
    'Bailey', 'Rivera', 'Cooper', 'Richardson', 'Cox', 'Howard', 'Ward', 'Torres', 'Peterson', 'Gray',
    'Ramirez', 'James', 'Watson', 'Brooks', 'Kelly', 'Sanders', 'Price', 'Bennett', 'Wood', 'Barnes',
    'Ross', 'Henderson', 'Coleman', 'Jenkins', 'Perry', 'Powell', 'Long', 'Patterson', 'Hughes', 'Flores',
    'Washington', 'Butler', 'Simmons', 'Foster', 'Gonzales', 'Bryant', 'Alexander', 'Russell', 'Griffin', 'Diaz'
];

const generateMockScholar = (index: number): Scholar => {
    const firstName = getRandomElement(firstNames);
    const lastName = getRandomElement(lastNames);
    const name = `Dr. ${firstName} ${lastName}`;
    const username = `@${firstName.toLowerCase()}${lastName.toLowerCase()}`;
    const avatar = firstName[0];
    const institution = getRandomElement(institutions);
    const department = getRandomElement(departments);
    const title = getRandomElement(titles);
    const expertise = getRandomElements(expertiseAreas, Math.floor(Math.random() * 4) + 2);
    const hIndex = Math.floor(Math.random() * 50) + 5;
    const totalCitations = Math.floor(Math.random() * 10000) + 100;
    const paperCount = Math.floor(Math.random() * 8) + 1;
    
    const papers = Array.from({ length: paperCount }, (_, i) => ({
        id: `arxiv:${2020 + Math.floor(Math.random() * 5)}.${Math.random().toString(36).substr(2, 8)}`,
        title: `${expertise[0]} for ${expertise[1]}: A Novel Approach`,
        year: 2020 + Math.floor(Math.random() * 5),
        citations: Math.floor(Math.random() * 500) + 10,
        url: `https://arxiv.org/abs/${Math.random().toString(36).substr(2, 8)}`
    }));

    const activities = ['2 days ago', '1 week ago', '3 days ago', '5 days ago', '1 day ago', '4 days ago'];
    const recentActivity = getRandomElement(activities);

    return {
        id: `${firstName.toLowerCase()}-${lastName.toLowerCase()}-${index}`,
        name,
        username,
        avatar,
        institution,
        department,
        title,
        bio: `Research focuses on ${expertise[0].toLowerCase()} and ${expertise[1].toLowerCase()}. Currently working on ${expertise[2]?.toLowerCase() || expertise[0].toLowerCase()} methods for ${expertise[3]?.toLowerCase() || 'advanced applications'}.`,
        expertise,
        hIndex,
        totalCitations,
        papers,
        recentActivity,
        isFollowing: Math.random() > 0.7,
        claimed: Math.random() > 0.3
    };
};

export const mockScholars: Scholar[] = Array.from({ length: 200 }, (_, index) => generateMockScholar(index)); 