export interface Scholar {
    id: string;
    name: string;
    username: string;
    avatar: string;
    institution: string;
    department: string;
    title: string;
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
    institutionId: string;
}

import { scholarsA_M } from './scholarsA_M';
import { scholarsN_Z } from './scholarsN_Z';

export const mockScholars = [...scholarsA_M, ...scholarsN_Z]; 