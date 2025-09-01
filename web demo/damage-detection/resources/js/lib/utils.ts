import {type ClassValue, clsx} from "clsx"
import {twMerge} from "tailwind-merge"

// import {Normalized} from "@/Pages/Scouts/types";

export function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs))
}

// export function denormalize<T>(data: Normalized<T>) {
//     return data.allIds.map(id => data.byId[id]);
// }
